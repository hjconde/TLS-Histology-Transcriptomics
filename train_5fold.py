#!/usr/bin/env python3
"""Train the multimodal TLS classifier with five-fold evaluation.

Select each fold's checkpoint using a validation split within its training
partition, then evaluate the corresponding held-out fold. All input files
are supplied locally under data/; no patient data or weights are bundled.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm.auto import tqdm

from tls_multimodal.model import FocalLoss
from tls_multimodal.data import (load_gene_matrix, validate_five_folds, validate_gene_embedding_array,
                                 load_fold_arrays, inner_train_val_split)
from tls_multimodal.engine import (set_seed, create_model, configure_finetuning,
                                   core_metrics, predict, checkpoint_model_state)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--gene-embeddings-npy", type=Path, default=None,
                   help="Locally prepared, gene-order-aligned data/gene_embeddings.npy")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--gene-order-file", type=Path, default=None,
                   help="Optional explicit gene order (one symbol per line)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-5,
                   help="Optimizer learning rate")
    p.add_argument("--weight-decay", type=float, default=2e-5)
    p.add_argument("--contrast-weight", type=float, default=0.2,
                   help="Coefficient for cross-modal contrastive loss")
    p.add_argument("--class-number", type=int, default=11)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--expected-patients", type=int, default=0,
                   help="Use 0 to disable patient-count check")
    p.add_argument("--expected-genes", type=int, default=0,
                   help="Use 0 to disable filtered-gene-count check")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def loader_for_subset(dataset, indices, labels, batch_size, device, seed):
    subset = Subset(dataset, list(map(int, indices)))
    y = np.asarray(labels, dtype=int)[indices]
    if len(set(y.tolist())) != 2:
        raise ValueError("Internal training subset must contain both TLS classes")
    class_counts = np.bincount(y, minlength=2)
    weights = torch.as_tensor([1.0 / class_counts[label] for label in y], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    # Inverse-class-frequency sampling is applied within the training partition.
    sampler = WeightedRandomSampler(weights, num_samples=len(subset),
                                    replacement=True, generator=generator)
    return DataLoader(subset, batch_size=batch_size, sampler=sampler,
                      num_workers=0, pin_memory=(device.type == "cuda")), y


def make_eval_loader(dataset, batch_size, device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=(device.type == "cuda"))


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, contrast_weight, accumulation):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for step, (gene, image, label) in enumerate(loader):
        gene = gene.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits, contrast = model(gene, image, labels=label,
                                     contrast_weight=contrast_weight)
            combined = criterion(logits, label) + contrast
            loss = combined / accumulation
        scaler.scale(loss).backward()
        if ((step + 1) % accumulation == 0) or (step + 1 == len(loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(combined.detach().cpu())
    return total_loss / len(loader)


def main():
    a = parse_args()
    if not 0 < a.val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    if a.train_batch_size < 2:
        raise ValueError("Contrastive training requires train batch size >= 2")
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          "cpu" if a.device == "auto" else a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(a.seed)
    print(f"Device={device}; random seed={a.seed}")
    print("Validation selects checkpoints; held-out folds are used for evaluation.")

    gene_csv = a.data_root / "TLS_gene_fold5" / "hj_scaler_mat.csv"
    gene2vec = a.data_root / "Gene2vec_0111.txt"
    pretrained_path = a.data_root / "GEO2KEGG_pretrain_02211552_27.pth"
    matrix = load_gene_matrix(gene_csv, gene2vec, a.class_number, a.gene_order_file)
    if a.expected_patients and len(matrix.sample_ids) != a.expected_patients:
        raise ValueError(f"Expected {a.expected_patients} patients, found {len(matrix.sample_ids)}")
    if a.expected_genes and len(matrix.gene_names) != a.expected_genes:
        raise ValueError(f"Expected {a.expected_genes} genes, found {len(matrix.gene_names)}")
    validate_five_folds(a.data_root / "TLS_gene_fold5", matrix)
    gene_embeddings_npy = a.gene_embeddings_npy or (a.data_root / "gene_embeddings.npy")
    validate_gene_embedding_array(gene_embeddings_npy, matrix.gene_names)
    os.environ["TLS_GENE2VEC_NPY"] = str(gene_embeddings_npy.resolve())

    run_dir = a.output_dir / f"validation_selected_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "gene_order.txt").write_text("\n".join(matrix.gene_names) + "\n", encoding="utf-8")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    (run_dir / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = []
    for fold in range(1, 6):
        set_seed(a.seed + fold)
        train_dataset, test_dataset = load_fold_arrays(a.data_root, matrix, fold)
        train_positions, val_positions = inner_train_val_split(
            train_dataset.labels, fold, a.seed, a.val_fraction
        )
        train_loader, train_y = loader_for_subset(
            train_dataset, train_positions, train_dataset.labels,
            a.train_batch_size, device, a.seed + fold
        )
        val_loader = make_eval_loader(Subset(train_dataset, val_positions), a.eval_batch_size, device)
        test_loader = make_eval_loader(test_dataset, a.eval_batch_size, device)
        print(f"Fold {fold}: inner train={len(train_positions)}, val={len(val_positions)}, "
              f"outer test={len(test_dataset)}")

        model = create_model(len(matrix.gene_names), a.class_number, pretrained_path).to(device)
        configure_finetuning(model)
        optimizer = Adam((p for p in model.parameters() if p.requires_grad),
                         lr=a.learning_rate, eps=1e-6, weight_decay=a.weight_decay)
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        if hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        else:  # Compatibility with PyTorch versions lacking torch.amp.GradScaler.
            scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        n_pos = int(np.sum(train_y == 1))
        n_neg = int(np.sum(train_y == 0))
        criterion = FocalLoss(alpha=torch.tensor([1.0, n_neg / n_pos],
                                                device=device), gamma=2)
        best = (-float("inf"), -float("inf"))
        best_epoch = None
        stale = 0
        fold_dir = run_dir / "ckpts" / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = fold_dir / "best_model_by_core_metrics.pth"
        for epoch in tqdm(range(a.epochs), desc=f"Fold {fold}"):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                                   scaler, device, a.contrast_weight, a.gradient_accumulation)
            (warmup if epoch < 10 else scheduler).step()
            # VALIDATION: subset of outer-training patients, not outer test patients.
            truth_val, prob_val = predict(model, val_loader, device)
            val_metrics = core_metrics(truth_val, prob_val)
            score = (val_metrics["auc"], val_metrics["balanced_acc"])
            if score > best:
                best = score
                best_epoch = epoch
                stale = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "fold": fold,
                    "selection": "inner_train_validation_auc",
                    "validation_metrics": val_metrics,
                    "num_genes": len(matrix.gene_names),
                    "class_number": a.class_number,
                }, checkpoint_path)
            else:
                stale += 1
            if epoch % 10 == 0 or stale >= a.patience:
                tqdm.write(f"Fold {fold} epoch {epoch}: loss={loss:.4f}, "
                           f"val_auc={val_metrics['auc']:.4f}, best_val_auc={best[0]:.4f}")
            if stale >= a.patience:
                break

        # TEST: accessed for the first and only time AFTER checkpoint selection.
        checkpoint = checkpoint_model_state(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        truth_test, prob_test = predict(model, test_loader, device)
        test_metrics = core_metrics(truth_test, prob_test)
        test_metrics.update({"fold": fold, "best_epoch": best_epoch,
                             "val_auc": best[0], "n_test": len(test_dataset)})
        rows.append(test_metrics)
        pred_dir = run_dir / "pred_logs"
        pred_dir.mkdir(parents=True, exist_ok=True)
        # Contains clinical labels: saved LOCALLY under ignored outputs/ only.
        pd.DataFrame({
            "sample_id": test_dataset.sample_ids,
            "truth_label": truth_test,
            "pred_label": np.argmax(prob_test, axis=1),
            "pred_prob_0": prob_test[:, 0],
            "pred_prob_1": prob_test[:, 1],
        }).to_csv(pred_dir / f"fold{fold}_test_pred_log.csv", index=False)
        pd.DataFrame([test_metrics]).to_csv(pred_dir / f"fold{fold}_core_metrics.csv", index=False)
        print(f"Fold {fold} FINAL held-out AUC={test_metrics['auc']:.6f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    report_cols = ["accuracy", "balanced_acc", "auc", "ppv", "npv"]
    summary = pd.concat([
        frame,
        pd.DataFrame([{"fold": "mean", **{col: frame[col].mean() for col in report_cols}}]),
        pd.DataFrame([{"fold": "std", **{col: frame[col].std(ddof=0) for col in report_cols}}]),
    ], ignore_index=True)
    summary.to_csv(run_dir / "5fold_core_metrics_summary.csv", index=False, float_format="%.6f")
    print("\nValidation-selected 5-fold test results:")
    print(summary[["fold"] + report_cols].to_string(index=False))
    print(f"\nAll outputs are local and gitignored: {run_dir}")


if __name__ == "__main__":
    main()
