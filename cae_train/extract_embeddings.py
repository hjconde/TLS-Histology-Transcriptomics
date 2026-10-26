"""Export 256-dimensional CAE image vectors for multimodal fusion.

Supports global-average pooling over 256-channel class-associated content
features or export of a 256-dimensional individual-specific style code.
Inputs and generated files are kept locally inside data/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .networks import CAEGen

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}


def normalized_id(s: str) -> str:
    return re.sub(r"\s+", "", s.replace("\u00a0", "").replace("\u3000", "")).upper()


def load_split_file(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing private patient split file: {path}")
    ids = []
    for row in path.read_text(encoding="utf-8").splitlines():
        if not row.strip():
            continue
        fields = row.split()
        if len(fields) < 2 or fields[1] not in {"0", "1"}:
            raise ValueError(f"Split file {path} must contain SAMPLE_ID and 0/1 label")
        ids.append(normalized_id(fields[0]))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Split file {path} is empty or contains duplicate sample IDs")
    return ids


def read_image_id_map(path: Path | None) -> Dict[str, str]:
    """Optional PRIVATE CSV with image_filename,sample_id for nonstandard names."""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Missing private filename mapping: {path}")
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image_filename", "sample_id"}.issubset(reader.fieldnames):
            raise ValueError("Image-ID mapping requires image_filename,sample_id header")
        for row in reader:
            key = row["image_filename"].strip()
            sid = normalized_id(row["sample_id"].strip())
            if not key or not sid or key in mapping:
                raise ValueError("Duplicate or blank filename/sample ID in private mapping")
            mapping[key] = sid
    return mapping


def image_to_tensor(file: Path, size: int = 256) -> torch.Tensor:
    """CAE train-time test preprocessing: resize, ToTensor, Normalize(0.5,0.5)."""
    with Image.open(file) as loaded:
        image = loaded.convert("RGB").resize((size, size), resample=Image.Resampling.BILINEAR)
        rgb = np.array(image, dtype=np.float32, copy=True)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().div_(255.).sub_(0.5).div_(0.5)


class PrivateImageDataset(Dataset):
    def __init__(self, image_dir: Path, expected_ids: List[str], mapping: Dict[str, str], size: int):
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Missing local, authorized image directory: {image_dir}")
        if not 0 < size:
            raise ValueError("Image size must be positive")
        files = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise ValueError(f"No supported JPG/PNG/BMP/PPM images found under {image_dir}")
        resolved = {}
        for path in files:
            sid = mapping.get(path.name, mapping.get(path.stem, normalized_id(path.stem)))
            if sid in resolved:
                raise ValueError(f"Multiple images resolve to a single patient ID; use a one-image-per-patient mapping")
            resolved[sid] = path
        expected_set = set(expected_ids)
        unknown = set(resolved) - expected_set
        missing = expected_set - set(resolved)
        if unknown or missing:
            raise ValueError(
                f"Input images and fold split do not match (unknown={len(unknown)}, missing={len(missing)}). "
                "Provide a private --image-id-map mapping image_filename,sample_id, or use patient IDs as filenames."
            )
        self.rows = [(sid, resolved[sid]) for sid in expected_ids]
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        sid, path = self.rows[index]
        return sid, image_to_tensor(path, self.size)


def load_generator(checkpoint_path: Path, *, style_dim: int, device: torch.device) -> CAEGen:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CAE generator checkpoint missing: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a checkpoint dictionary containing the 'ab' generator weights")
    weights = checkpoint.get("ab", checkpoint.get("model_state_dict"))
    if weights is None:
        raise KeyError("CAE checkpoint must contain 'ab' or 'model_state_dict' weights")
    model = CAEGen(style_dim=style_dim)
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "CAE checkpoint incompatible with selected --style-dim. "
            "Choose --style-dim to match the checkpoint architecture."
        ) from exc
    return model.to(device).eval()


def encode_256(model: CAEGen, images: torch.Tensor, method: str) -> torch.Tensor:
    """Return one 256-D vector per image."""
    content, style = model.encode(images)
    if method == "content_gap":
        # [B, 256, H, W] -> [B, 256]. Preserve class-related/content channels.
        features = F.adaptive_avg_pool2d(content, output_size=1).flatten(1)
    elif method == "style":
        # 256-dimensional individual-specific/style code, NOT class-associated.
        features = style.flatten(1)
    else:
        raise ValueError(f"Unknown method: {method}")
    if features.ndim != 2 or features.shape[1] != 256:
        raise ValueError(
            f"Selected features have shape {tuple(features.shape)} instead of [N,256]. "
            "For style embeddings, train/load a CAE with style_dim=256."
        )
    if not torch.isfinite(features).all():
        raise ValueError("CAE encoder produced non-finite feature values")
    return features


@torch.inference_mode()
def export_partition(model: CAEGen, dataset: Dataset, output_file: Path,
                     *, method: str, device: torch.device, batch_size: int) -> int:
    if batch_size < 1:
        raise ValueError("--batch-size must be >=1")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    temp_path = output_file.with_suffix(output_file.suffix + ".tmp")
    n_written = 0
    try:
        with temp_path.open("w", encoding="utf-8") as out:
            for ids, images in loader:
                encoded = encode_256(model, images.to(device), method).cpu().numpy()
                for sid, row in zip(ids, encoded):
                    values = " ".join(f"{float(x):.9g}" for x in row)
                    out.write(f"{sid} latent: [{values}]\n")
                    n_written += 1
        temp_path.replace(output_file)
    finally:
        temp_path.unlink(missing_ok=True)
    return n_written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Export local 256-D CAE image features, with strict patient-fold checks"
    )
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6), metavar="1..5")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Private fold-specific CAE gen_XXXXXXXX.pt checkpoint")
    parser.add_argument("--train-image-dir", type=Path, required=True,
                        help="Pre-rendered training H&E raster images; may include subfolders")
    parser.add_argument("--test-image-dir", type=Path, required=True,
                        help="Pre-rendered held-out H&E raster images; may include subfolders")
    parser.add_argument("--split-dir", type=Path, default=Path("data/TLS_gene_fold5"))
    parser.add_argument("--image-id-map", type=Path, default=None,
                        help="Optional PRIVATE CSV image_filename,sample_id (not to be published)")
    parser.add_argument("--output-root", type=Path, default=Path("data/hjImage"))
    parser.add_argument("--style-dim", type=int, default=256,
                        help="CAE checkpoint's individual-specific code size (default 256)")
    parser.add_argument("--method", choices=["content_gap", "style"], default="content_gap",
                        help="content_gap: class-associated feature pooling; style: individual-specific code")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-tag", default="gen_00004000.pt",
                        help="Used solely to construct the conventional downstream output filenames")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but not available")
    train_ids = load_split_file(args.split_dir / f"train_fold{args.fold}.txt")
    test_ids = load_split_file(args.split_dir / f"test_fold{args.fold}.txt")
    if set(train_ids) & set(test_ids):
        raise ValueError("Patient overlap between the training and test partitions")
    mapping = read_image_id_map(args.image_id_map)
    train = PrivateImageDataset(args.train_image_dir, train_ids, mapping, args.size)
    test = PrivateImageDataset(args.test_image_dir, test_ids, mapping, args.size)
    model = load_generator(args.checkpoint, style_dim=args.style_dim, device=device)
    # Use the filename convention expected by the fusion data loader.
    tag = Path(args.checkpoint_tag).name
    if "/" in args.checkpoint_tag or "\\" in args.checkpoint_tag or not re.fullmatch(r"gen_\d{8}\.pt", tag):
        raise ValueError("--checkpoint-tag must look like gen_00004000.pt")
    destination = args.output_root / f"fold{args.fold}_latent_all_model"
    output_train = destination / f"trainAB_latent_e_{tag}_s256_new2.txt"
    output_test = destination / f"testAB_latent_e_{tag}_s256_new2.txt"
    for path in [output_train, output_test]:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing feature file: {path}. "
                "Move/rename it before exporting."
            )
    n_train = export_partition(model, train, output_train, method=args.method,
                               device=device, batch_size=args.batch_size)
    n_test = export_partition(model, test, output_test, method=args.method,
                              device=device, batch_size=args.batch_size)
    (destination / "FEATURE_EXTRACTION_METHOD.json").write_text(
        json.dumps({
            "method": args.method,
            "feature_dimension": 256,
            "style_dimension": args.style_dim,
            "input_size": [args.size, args.size],
            "fold": args.fold,
            "train_count": n_train,
            "test_count": n_test
        }, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Fold {args.fold}: exported train={n_train}, test={n_test} vectors of length 256")
    print(f"Outputs are PRIVATE and gitignored: {destination}")


if __name__ == "__main__":
    main()
