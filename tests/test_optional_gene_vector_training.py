"""Synthetic corpus test; avoids any patient gene inputs."""
from pathlib import Path
import pytest


def test_gene2vec_training_on_synthetic_gene_pairs(tmp_path):
    gensim = pytest.importorskip("gensim")
    from gene2vec.gene2vec import load_source, gene_vectorization
    from gene2vec.generateMatrix import load_embeddings
    inp = tmp_path / "corpus"
    inp.mkdir()
    (inp / "data.txt").write_text("GENEA GENEB\nGENEB GENEC\nGENEA GENEC\n")
    corpus = load_source(inp, 'txt')
    model_path = gene_vectorization(corpus, tmp_path / "models", dimension=200,
                                    workers=1, window=1, iterations=2,
                                    sg=1, seed=123, write_text=True)
    vectors, genes = load_embeddings(model_path)
    assert vectors.shape == (3, 200)
    assert set(genes) == {'GENEA', 'GENEB', 'GENEC'}
    assert (Path(str(model_path) + '.txt')).is_file()
