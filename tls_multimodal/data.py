"""Load authorized, locally supplied encoded expression, fold labels and CAE latents.

ALL patient-level input files must remain inside the ignored data directory.
No patient or TLS-label data is distributed with the public repository.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def normalize_sample_id(sample_id: object) -> str:
    """Normalize sample identifiers for modality joins."""
    if pd.isna(sample_id):
        return ""
    return re.sub(r"\s+", "", str(sample_id).replace("\u00a0", "").replace("\u3000", "")).upper()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required local file not found: {path}\n"
            "Place your authorized data under data/; never commit it to GitHub."
        )
    return path


@dataclass
class GeneMatrix:
    expressions: np.ndarray  # patient x gene, integer token IDs produced from input values
    sample_ids: List[str]
    gene_names: List[str]
    id_to_row: Dict[str, int]


def load_gene_matrix(gene_csv: Path, gene_embedding_txt: Path,
                     class_number: int = 11, gene_order_file: Path | None = None) -> GeneMatrix:
    """Load preprocessed expression values as gene-expression tokens.

    Values above class_number are clipped and nonnegative fractional values
    are truncated to integers. Only genes with Gene2Vec vectors are retained.
    Column order follows the matrix or an explicitly supplied gene list.
    """
    frame = pd.read_csv(require_file(gene_csv), index_col=0)
    if frame.empty:
        raise ValueError("Gene matrix is empty")
    sample_ids = [normalize_sample_id(idx) for idx in frame.index]
    if any(not sid for sid in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Gene matrix has empty or repeated normalized sample IDs")
    emb_names = pd.read_csv(
        require_file(gene_embedding_txt), sep=r"\s+", header=None, usecols=[0]
    ).iloc[:, 0].astype(str).tolist()
    known = set(emb_names)
    common = [str(c) for c in frame.columns if str(c) in known]
    if gene_order_file is not None:
        exact_order = [s.strip() for s in require_file(gene_order_file).read_text().splitlines() if s.strip()]
        if len(exact_order) != len(set(exact_order)) or set(exact_order) != set(common):
            raise ValueError("gene_order.txt must contain every overlapping gene exactly once")
        common = exact_order
    if not common:
        raise ValueError("No shared gene names between expression matrix and Gene2vec")
    values = frame.loc[:, common].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite expression values; supply fully processed tokenized input")
    if np.any(values < 0):
        raise ValueError(
            "Negative input values would produce invalid gene-token indices. "
            "Provide the same nonnegative study-preprocessed matrix as used in training."
        )
    # Clip to the configured token maximum and truncate nonnegative fractions.
    values = np.minimum(values, class_number).astype(np.int64)
    return GeneMatrix(values, sample_ids, common, {sid: k for k, sid in enumerate(sample_ids)})


def validate_gene_embedding_array(path: Path, expected_gene_names: List[str],
                                  dim: int = 200) -> None:
    """Validate private .npy and, when available, its ordered-gene manifest."""
    import warnings
    arr = np.load(require_file(path), allow_pickle=False, mmap_mode='r')
    expected = (len(expected_gene_names), dim)
    if arr.shape != expected:
        raise ValueError(f"Gene2Vec matrix shape {arr.shape} does not match {expected}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Nonfinite gene embeddings in {path}")
    manifest = path.with_name(path.stem + '.genes.txt')
    if manifest.is_file():
        lines = [row.strip() for row in manifest.read_text(encoding='utf-8').splitlines()]
        if lines != expected_gene_names:
            raise ValueError(
                "Gene2Vec vector order does not match expression gene order; "
                "rerun prepare_gene_embeddings.py with the correct --gene-order-file."
            )
    else:
        warnings.warn(
            "Gene2Vec order manifest missing; vector ordering cannot be verified. "
            "Generate it using prepare_gene_embeddings.py, or supply an ordered manifest.",
            RuntimeWarning
        )


def read_fold_labels(path: Path) -> List[Tuple[str, int]]:
    rows = []
    for i, line in enumerate(require_file(path).read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 2:
            raise ValueError(f"Malformed label line {i} in {path}")
        sid = normalize_sample_id(fields[0])
        label = int(fields[1])
        if label not in (0, 1):
            raise ValueError(f"Expected 0/1 TLS labels in {path}, line {i}")
        rows.append((sid, label))
    if not rows or len(rows) != len({sid for sid, _ in rows}):
        raise ValueError(f"Fold file is empty or contains duplicate patients: {path}")
    return rows


def validate_five_folds(split_dir: Path, matrix: GeneMatrix) -> None:
    test_seen = set()
    labels_seen: Dict[str, int] = {}
    for k in range(1, 6):
        train = read_fold_labels(split_dir / f"train_fold{k}.txt")
        test = read_fold_labels(split_dir / f"test_fold{k}.txt")
        train_ids = {sid for sid, _ in train}
        test_ids = {sid for sid, _ in test}
        if train_ids & test_ids:
            raise ValueError(f"Patient overlap between train and test in fold {k}")
        if test_ids & test_seen:
            raise ValueError(f"One patient appears in more than one outer test fold (fold {k})")
        test_seen |= test_ids
        for sid, lab in train + test:
            if sid not in matrix.id_to_row:
                raise ValueError(f"Fold patient {sid} missing in gene input")
            if sid in labels_seen and labels_seen[sid] != lab:
                raise ValueError("Conflicting TLS labels across fold files")
            labels_seen[sid] = lab
    if test_seen != set(matrix.sample_ids):
        missing = set(matrix.sample_ids) - test_seen
        extra = test_seen - set(matrix.sample_ids)
        raise ValueError(f"Outer test folds must cover each input patient once; missing={len(missing)}, extra={len(extra)}")


def load_image_embedding(path: Path, expected_dim: int = 256) -> Dict[str, np.ndarray]:
    """Read `SAMPLE_ID latent: [...]` lines from the supplied CAE feature files."""
    result: Dict[str, np.ndarray] = {}
    for line_num, line in enumerate(require_file(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if "latent:" not in line:
            raise ValueError(f"Missing 'latent:' in {path} line {line_num}")
        raw_id, latent = line.split("latent:", 1)
        sid = normalize_sample_id(raw_id)
        numbers = [float(m.group()) for m in _FLOAT_PATTERN.finditer(latent)]
        if len(numbers) != expected_dim or sid in result:
            raise ValueError(f"Invalid or duplicate {expected_dim}-D feature for {sid} in {path}:{line_num}")
        arr = np.asarray(numbers, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite image feature for {sid}")
        result[sid] = arr
    return result


def load_fold_arrays(data_root: Path, matrix: GeneMatrix, fold: int):
    train_rows = read_fold_labels(data_root / "TLS_gene_fold5" / f"train_fold{fold}.txt")
    test_rows = read_fold_labels(data_root / "TLS_gene_fold5" / f"test_fold{fold}.txt")
    latent_dir = data_root / "hjImage" / f"fold{fold}_latent_all_model"
    train_img = load_image_embedding(latent_dir / "trainAB_latent_e_gen_00004000.pt_s256_new2.txt")
    test_img = load_image_embedding(latent_dir / "testAB_latent_e_gen_00004000.pt_s256_new2.txt")

    def assemble(rows, images, name):
        missing = [sid for sid, _ in rows if sid not in images]
        if missing:
            raise ValueError(f"Fold {fold}: {len(missing)} {name} patients lack CAE embeddings")
        indices = [matrix.id_to_row[sid] for sid, _ in rows]
        xgene = matrix.expressions[indices]
        ximage = np.stack([images[sid] for sid, _ in rows]).astype(np.float32)
        y = np.asarray([lab for _, lab in rows], dtype=np.int64)
        return GeneImageDataset(xgene, ximage, y, [sid for sid, _ in rows])
    return assemble(train_rows, train_img, "train"), assemble(test_rows, test_img, "test")


class GeneImageDataset(Dataset):
    def __init__(self, genes: np.ndarray, image_latents: np.ndarray,
                 labels: np.ndarray, sample_ids: List[str]):
        if len(genes) != len(image_latents) or len(genes) != len(labels):
            raise ValueError("Gene, image and label arrays must be aligned by patient")
        if image_latents.ndim != 2 or image_latents.shape[1] != 256:
            raise ValueError("Expected one 256-dimensional CAE latent per patient")
        self.genes = np.concatenate([genes.astype(np.int64),
                    np.zeros((len(genes), 1), dtype=np.int64)], axis=1)
        self.images = image_latents.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (torch.from_numpy(self.genes[index]),
                torch.from_numpy(self.images[index]).unsqueeze(0),
                torch.tensor(self.labels[index], dtype=torch.long))


def inner_train_val_split(labels: np.ndarray, fold: int, seed: int,
                          val_fraction: float = 0.2):
    """The split is made using OUTER-TRAIN patients ONLY."""
    ids = np.arange(len(labels))
    return train_test_split(ids, test_size=val_fraction,
                            random_state=seed + fold, stratify=labels)
