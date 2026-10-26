"""Synthetic-only tests of the newly written CAE feature exporter."""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cae_train.networks import CAEGen
from cae_train.extract_embeddings import (
    PrivateImageDataset, encode_256, image_to_tensor, load_generator,
    load_split_file, export_partition, main,
)
from tls_multimodal.data import load_image_embedding


def test_content_gap_and_256_style_shapes():
    model = CAEGen(style_dim=256).eval()
    with torch.inference_mode():
        x = torch.zeros((1, 3, 32, 32))
        content = encode_256(model, x, "content_gap")
        style = encode_256(model, x, "style")
    assert content.shape == (1, 256)
    assert style.shape == (1, 256)
    assert torch.isfinite(content).all()


def test_private_split_and_image_id_guard(tmp_path: Path):
    p = tmp_path / "test_fold1.txt"
    p.write_text("SYNTHETIC_PATIENT1 0\nSYNTHETIC_PATIENT2 1\n")
    assert load_split_file(p) == ["SYNTHETIC_PATIENT1", "SYNTHETIC_PATIENT2"]
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
        images / "SYNTHETIC_PATIENT1.png")
    with pytest.raises(ValueError, match="do not match"):
        PrivateImageDataset(images, load_split_file(p), {}, 32)


def test_checkpoint_to_private_latents(tmp_path: Path):
    # A randomly initialized synthetic model tests code wiring, not science.
    ckpt = tmp_path / "synthetic_gen_00004000.pt"
    model = CAEGen(style_dim=256).eval()
    torch.save({"ab": model.state_dict()}, ckpt)
    del model

    split = tmp_path / "TLS_gene_fold5"
    split.mkdir()
    (split / "train_fold1.txt").write_text("SYNTHETIC1 0\n", encoding="utf-8")
    (split / "test_fold1.txt").write_text("SYNTHETIC2 1\n", encoding="utf-8")
    train = tmp_path / "train"
    test = tmp_path / "test"
    train.mkdir()
    test.mkdir()
    Image.fromarray(np.full((32, 32, 3), 70, dtype=np.uint8)).save(train / "SYNTHETIC1.png")
    Image.fromarray(np.full((32, 32, 3), 170, dtype=np.uint8)).save(test / "SYNTHETIC2.png")
    output = tmp_path / "data" / "hjImage"
    main([
        "--fold", "1",
        "--checkpoint", str(ckpt),
        "--train-image-dir", str(train),
        "--test-image-dir", str(test),
        "--split-dir", str(split),
        "--output-root", str(output),
        "--batch-size", "1",
        "--size", "32",
        "--device", "cpu",
    ])
    dest = output / "fold1_latent_all_model"
    train_vectors = load_image_embedding(dest / "trainAB_latent_e_gen_00004000.pt_s256_new2.txt")
    test_vectors = load_image_embedding(dest / "testAB_latent_e_gen_00004000.pt_s256_new2.txt")
    assert train_vectors["SYNTHETIC1"].shape == (256,)
    assert test_vectors["SYNTHETIC2"].shape == (256,)
    assert (dest / "FEATURE_EXTRACTION_METHOD.json").is_file()
    with pytest.raises(FileExistsError):
        main([
            "--fold", "1", "--checkpoint", str(ckpt),
            "--train-image-dir", str(train), "--test-image-dir", str(test),
            "--split-dir", str(split), "--output-root", str(output),
            "--size", "32", "--device", "cpu",
        ])
