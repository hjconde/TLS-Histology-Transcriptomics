#!/usr/bin/env python3
"""Load locally supplied five-fold checkpoints and evaluate predictions.

This program performs inference only and writes outputs to the local
ignored outputs/ directory. It does not distribute input data or weights.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tls_multimodal.data import load_gene_matrix, validate_five_folds, validate_gene_embedding_array, load_fold_arrays
from tls_multimodal.engine import (create_model, checkpoint_model_state,
                                   predict, core_metrics, set_seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--gene-embeddings-npy", type=Path, default=None,
                   help="Locally prepared, gene-order-aligned data/gene_embeddings.npy")
    p.add_argument("--checkpoints", type=Path, required=True,
                   help="LOCAL folder containing ckpts/fold1/ ... ckpts/fold5/")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    p.add_argument("--gene-order-file", type=Path, default=None)
    p.add_argument("--class-number", type=int, default=11)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    matrix = load_gene_matrix(args.data_root / "TLS_gene_fold5" / "hj_scaler_mat.csv",
                              args.data_root / "Gene2vec_0111.txt",
                              args.class_number, args.gene_order_file)
    validate_five_folds(args.data_root / "TLS_gene_fold5", matrix)
    gene_embeddings_npy = args.gene_embeddings_npy or (args.data_root / "gene_embeddings.npy")
    validate_gene_embedding_array(gene_embeddings_npy, matrix.gene_names)
    os.environ["TLS_GENE2VEC_NPY"] = str(gene_embeddings_npy.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    out_of_fold = []
    for fold in range(1, 6):
        _, test_dataset = load_fold_arrays(args.data_root, matrix, fold)
        path = args.checkpoints / "ckpts" / f"fold{fold}" / "best_model_by_core_metrics.pth"
        if not path.is_file():
            # Also accept ckpts/ as the --checkpoints root.
            path = args.checkpoints / f"fold{fold}" / "best_model_by_core_metrics.pth"
        checkpoint = checkpoint_model_state(path)
        if checkpoint.get("num_genes", len(matrix.gene_names)) != len(matrix.gene_names):
            raise ValueError(f"Fold {fold} checkpoint gene count mismatch")
        # The full architecture comes from the genuine installed gaBERT module.
        # We do not need to load an external pretrained checkpoint when a full
        # fold model state is available.
        model = create_model(len(matrix.gene_names), args.class_number,
                             pretrained_path=None, load_pretrained=False).to(device)
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Fold {fold}: saved weights are incompatible with the "
                "current model or gene ordering. Check the bundled "
                "performer_pytorch module and supplied gene order."
            ) from exc
        loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                            shuffle=False, num_workers=0)
        y, prob = predict(model, loader, device)
        row = {"fold": fold, **core_metrics(y, prob)}
        metrics.append(row)
        out_of_fold.append(pd.DataFrame({
            "sample_id": test_dataset.sample_ids,
            "truth_label": y,
            "pred_label": np.argmax(prob, axis=1),
            "pred_prob_0": prob[:, 0],
            "pred_prob_1": prob[:, 1],
        }))
        print(f"Fold {fold}: AUC={row['auc']:.6f}; checkpoint epoch={checkpoint.get('epoch', 'unknown')}")

    frame = pd.DataFrame(metrics)
    cols = ["accuracy", "balanced_acc", "auc", "ppv", "npv"]
    summary = pd.concat([
        frame,
        pd.DataFrame([{"fold": "mean", **{col: frame[col].mean() for col in cols}}]),
        pd.DataFrame([{"fold": "std", **{col: frame[col].std(ddof=0) for col in cols}}]),
    ], ignore_index=True)
    summary.to_csv(args.output_dir / "5fold_core_metrics_summary.csv", index=False)
    # OOF predictions contain patient labels: keep only in gitignored outputs/!
    pd.concat(out_of_fold, ignore_index=True).to_csv(
        args.output_dir / "out_of_fold_predictions_PRIVATE.csv", index=False
    )
    print(summary[["fold"] + cols].to_string(index=False))
    print(f"Private local outputs saved to {args.output_dir}")
    print("Evaluation finished; output files remain local.")


if __name__ == "__main__":
    main()
