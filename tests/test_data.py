from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tls_multimodal.data import (load_gene_matrix, load_image_embedding,
                                 read_fold_labels, validate_five_folds,
                                 inner_train_val_split)
from tls_multimodal.engine import core_metrics


def test_integer_data_and_folds(tmp_path):
    patient_ids = [f"TEST{i}" for i in range(10)]
    mat = pd.DataFrame({"GENEA": np.arange(10) % 12,
                        "GENEB": (np.arange(10) + 2) % 12}, index=patient_ids)
    mat.to_csv(tmp_path / "gene.csv")
    (tmp_path / "gene2vec.txt").write_text("GENEA 0.1 0.2\nGENEB 0.3 0.4\n")
    genes = load_gene_matrix(tmp_path / "gene.csv", tmp_path / "gene2vec.txt")
    assert genes.expressions.shape == (10, 2)
    for fold in range(1, 6):
        tests = patient_ids[(fold - 1) * 2:fold * 2]
        trains = [i for i in patient_ids if i not in tests]
        (tmp_path / f"test_fold{fold}.txt").write_text(
            "".join(f"{i} {int(i[-1]) % 2}\n" for i in tests))
        (tmp_path / f"train_fold{fold}.txt").write_text(
            "".join(f"{i} {int(i[-1]) % 2}\n" for i in trains))
    validate_five_folds(tmp_path, genes)
    i_train, i_val = inner_train_val_split(np.array([0, 1] * 4), 1, 2024, 0.25)
    assert set(i_train).isdisjoint(i_val)
    assert len(i_train) + len(i_val) == 8


def test_cae_feature_parser(tmp_path):
    path = tmp_path / "latents.txt"
    path.write_text("TESTA latent: [" + ", ".join("1e-3" for _ in range(256)) + "]\n")
    latents = load_image_embedding(path)
    assert latents["TESTA"].shape == (256,)


def test_fractional_tokens_follow_original_truncation(tmp_path):
    pd.DataFrame({"GENEA": [0.23, 2.75, 50.0]}, index=["A", "B", "C"]).to_csv(tmp_path / "mat.csv")
    (tmp_path / "embedding.txt").write_text("GENEA 0.5\n")
    result = load_gene_matrix(tmp_path / "mat.csv", tmp_path / "embedding.txt")
    assert result.expressions[:, 0].tolist() == [0, 2, 11]


def test_metric_definition():
    scores = core_metrics([0, 1, 0, 1], np.array([[0.9, 0.1], [0.2, 0.8],
                                                   [0.8, 0.2], [0.1, 0.9]]))
    assert all(scores[k] == 1 for k in ["accuracy", "balanced_acc", "auc", "ppv", "npv"])
