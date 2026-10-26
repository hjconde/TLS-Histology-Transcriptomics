#!/usr/bin/env python3
"""Prepare a gene-order-aligned (private) Gene2Vec matrix from local inputs.

This does NOT train Gene2Vec and does NOT contain patients/labels.
It reads the authorized original Gene2vec_0111.txt and aligns vectors
to the gene sequence used by the multimodal gene matrix.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from tls_multimodal.data import load_gene_matrix


def build_embeddings(gene_csv: Path, embedding_txt: Path, output_npy: Path,
                     gene_order_file: Path | None = None, vector_dim: int = 200) -> list[str]:
    matrix = load_gene_matrix(gene_csv, embedding_txt, gene_order_file=gene_order_file)
    target_genes = set(matrix.gene_names)
    known = {}
    with embedding_txt.open('r', encoding='utf-8') as infile:
        for line_no, line in enumerate(infile, 1):
            fields = line.split()
            if not fields:
                continue
            gene = fields[0]
            if gene not in target_genes:
                continue
            if gene in known:
                raise ValueError(f"Gene2Vec contains duplicate gene {gene!r} at line {line_no}")
            if len(fields) != vector_dim + 1:
                raise ValueError(
                    f"Gene {gene!r}: expected {vector_dim} coefficients, "
                    f"found {len(fields)-1} at line {line_no}"
                )
            known[gene] = np.asarray([float(v) for v in fields[1:]], dtype=np.float32)
    missing = [g for g in matrix.gene_names if g not in known]
    if missing:
        raise ValueError(f"Missing {len(missing)} genes from embedding file: {missing[:5]}")
    aligned = np.stack([known[g] for g in matrix.gene_names]).astype(np.float32)
    if not np.isfinite(aligned).all():
        raise ValueError("Gene2Vec vectors contain NaN/Infinity")
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, aligned, allow_pickle=False)
    names_path = output_npy.with_name(output_npy.stem + '.genes.txt')
    names_path.write_text('\n'.join(matrix.gene_names)+'\n', encoding='utf-8')
    print(f"Gene2Vec matrix saved to {output_npy}: {aligned.shape}")
    print(f"Gene order manifest saved to {names_path}")
    print("Embedding files have been written to the selected local data directory.")
    return matrix.gene_names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--gene-order-file', type=Path, default=None,
                   help='Optional original fold/model gene order; one gene per line')
    p.add_argument('--vector-dim', type=int, default=200)
    p.add_argument('--output', type=Path, default=None)
    args = p.parse_args()
    output = args.output or args.data_root / 'gene_embeddings.npy'
    build_embeddings(args.data_root/'TLS_gene_fold5'/'hj_scaler_mat.csv',
                     args.data_root/'Gene2vec_0111.txt', output,
                     gene_order_file=args.gene_order_file, vector_dim=args.vector_dim)


if __name__ == '__main__':
    main()
