"""Synthetic Gene2Vec-path, conversion, gene-order tests. No biomedical data."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from prepare_gene_embeddings import build_embeddings
from gene2vec.gene2vec_positional_embedding import Gene2VecPositionalEmbedding
from tls_multimodal.data import validate_gene_embedding_array


def test_gene2vec_data_conversion_and_frozen_embedding(tmp_path, monkeypatch):
    genes = ["GENE_B", "GENE_A", "GENE_C"]
    csv = tmp_path / "gene.csv"
    pd.DataFrame([[1, 2, 3], [5, 4, 2]], index=["FAKE1", "FAKE2"], columns=genes).to_csv(csv)
    txt = tmp_path / "genevec.txt"
    # Deliberately choose order unlike expression CSV, to exercise name alignment.
    with txt.open("w", encoding="utf-8") as f:
        for gene, v in [("GENE_A", 2), ("GENE_C", 3), ("GENE_B", 1)]:
            f.write(gene + " " + " ".join(map(str, [v] * 200)) + "\n")
    out = tmp_path / "gene_embeddings.npy"
    assert build_embeddings(csv, txt, out) == genes
    validate_gene_embedding_array(out, genes)
    monkeypatch.setenv("TLS_GENE2VEC_NPY", str(out))
    emb = Gene2VecPositionalEmbedding(dim=200, max_seq_len=4)
    x = emb(torch.zeros(2, 4, 200))
    assert x.shape == (4, 200)
    np.testing.assert_allclose(x[0].numpy(), np.ones(200))
    np.testing.assert_allclose(x[1].numpy(), np.full(200, 2))
    np.testing.assert_allclose(x[2].numpy(), np.full(200, 3))
    np.testing.assert_allclose(x[3].numpy(), np.zeros(200))
    assert "emb.weight" in emb.state_dict()
    assert not emb.emb.weight.requires_grad


def test_gene2vec_order_detection(tmp_path):
    p = tmp_path / "gene_embeddings.npy"
    np.save(p, np.zeros((3, 200), dtype=np.float32))
    p.with_name("gene_embeddings.genes.txt").write_text("A\nB\nC\n")
    validate_gene_embedding_array(p, ["A", "B", "C"])
    with pytest.raises(ValueError, match="order"):
        validate_gene_embedding_array(p, ["B", "A", "C"])
    with pytest.raises(ValueError, match="shape"):
        validate_gene_embedding_array(p, ["A", "B"])
