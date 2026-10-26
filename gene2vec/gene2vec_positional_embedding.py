"""Project-specific gaBERT Gene2Vec position embeddings.

Reads a locally prepared `.npy` file and returns pretrained positional
gene vectors. Uses a frozen `nn.Embedding` matrix.
Generate the array with prepare_gene_embeddings.py; do not commit it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Gene2VecPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        source = Path(os.environ.get(
            "TLS_GENE2VEC_NPY", "data/gene_embeddings.npy"
        )).expanduser()
        if not source.is_file():
            raise FileNotFoundError(
                f"Private Gene2Vec embedding matrix not found: {source}. "
                "Prepare it locally using `python prepare_gene_embeddings.py` "
                "or specify `--gene-embeddings-npy` to training/evaluation."
            )
        weights = np.load(source, allow_pickle=False)
        expected = (max_seq_len - 1, dim)
        if weights.ndim != 2 or tuple(weights.shape) != expected:
            raise ValueError(
                f"Gene2Vec embeddings have shape {weights.shape}, "
                f"expected {expected}; check gene order, count and vector size."
            )
        if not np.isfinite(weights).all():
            raise ValueError(f"Nonfinite Gene2Vec vector in {source}")
        padded = np.concatenate(
            (weights, np.zeros((1, dim), dtype=weights.dtype)), axis=0
        )
        # Keep the embedding attribute stable for checkpoint loading.
        self.emb = nn.Embedding.from_pretrained(
            torch.from_numpy(np.ascontiguousarray(padded)), freeze=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(x.shape[1], device=x.device)
        return self.emb(indices)
