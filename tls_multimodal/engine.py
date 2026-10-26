"""Model creation, fold-safe evaluation, metrics and checkpoint loading."""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

from .model import EnhancedImageEncoder, CrossModalAttentionFusion, GeneImageMultimodalModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_torch_load(path: Path, device: str | torch.device = "cpu"):
    """Only load model files obtained from trusted sources."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # For older supported PyTorch versions.
        return torch.load(path, map_location=device)


def make_gene_encoder(n_genes: int, n_tokens: int, gene_pos_emb=True) -> nn.Module:
    try:
        from performer_pytorch import PerformerLM
    except ImportError as exc:
        raise ImportError(
            "Missing the project-compatible performer_pytorch/ module. "
            "The research-specific source is bundled under performer_pytorch/. "
            "Do not substitute a generic pip version; it may not support "
            "g2v_position_emb and the saved gaBERT weights."
        ) from exc
    try:
        return PerformerLM(
            num_tokens=n_tokens, max_seq_len=n_genes + 1,
            dim=200, depth=6, heads=4, local_attn_heads=0,
            g2v_position_emb=gene_pos_emb,
        )
    except TypeError as exc:
        raise RuntimeError(
            "Incompatible PerformerLM constructor; requires "
            "the g2v_position_emb interface."
        ) from exc


def load_pretrained_gene_weights(gene_model: nn.Module, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained gaBERT checkpoint not supplied: {path}")
    checkpoint = safe_torch_load(path, "cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("Pretrained checkpoint has no state_dict")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    target = gene_model.state_dict()
    # Align positional embedding dimensions when needed.
    key = "pos_emb.emb.weight"
    if key in state and key in target and state[key].shape != target[key].shape:
        if state[key].ndim == 2 and target[key].ndim == 2 and state[key].shape[1] == target[key].shape[1]:
            new_value = target[key].clone()
            common = min(len(state[key]), len(new_value))
            new_value[:common] = state[key][:common]
            state[key] = new_value
        else:
            raise ValueError(f"Pretrained position embedding mismatch: {state[key].shape} vs {target[key].shape}")
    matches = [k for k, v in state.items() if k in target and tuple(v.shape) == tuple(target[k].shape)]
    conflicts = [k for k, v in state.items() if k in target and tuple(v.shape) != tuple(target[k].shape)]
    if conflicts:
        raise ValueError(f"Incompatible pretrained gaBERT weights: {conflicts[:5]}")
    fraction = len(matches) / max(len(target), 1)
    if fraction < 0.5:
        raise ValueError(
            f"Only {len(matches)}/{len(target)} gene-encoder parameter keys match the checkpoint. "
            "Refusing to continue with a mostly randomly initialized gaBERT."
        )
    missing, unexpected = gene_model.load_state_dict(state, strict=False)
    print(f"Pretrained gaBERT matched {len(matches)}/{len(target)} parameter keys; "
          f"missing={len(missing)}, unexpected={len(unexpected)}")


def create_model(n_genes: int, class_number: int, pretrained_path: Path | None,
                 *, load_pretrained=True) -> GeneImageMultimodalModel:
    gene = make_gene_encoder(n_genes, class_number + 2)
    if load_pretrained:
        if pretrained_path is None:
            raise ValueError("A matching pretrained gaBERT checkpoint is required")
        load_pretrained_gene_weights(gene, pretrained_path)
    image = EnhancedImageEncoder(img_dim=256, hidden_dim=256, out_dim=200)
    fusion = CrossModalAttentionFusion(gene_dim=200, img_dim=200, out_dim=200,
                                       num_heads=4, dropout_rate=0.1, skip=True)
    return GeneImageMultimodalModel(gene, image, fusion, num_classes=2, gene_out_dim=200)


def configure_finetuning(model: GeneImageMultimodalModel):
    # Fine-tune the last Performer block and normalization parameters.
    for p in model.gene_encoder.parameters():
        p.requires_grad = False
    unfrozen = []
    for name, p in model.gene_encoder.named_parameters():
        if "norm" in name or "performer.5" in name or "performer.net.layers.5." in name:
            p.requires_grad = True
            unfrozen.append(name)
    if not unfrozen:
        raise RuntimeError("No gaBERT parameters were selected for fine-tuning; check your PerformerLM version")
    print(f"Fine-tuning {len(unfrozen)} gaBERT parameter tensors (norms/final block)")


def core_metrics(y_true, probabilities):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape != (len(y), 2):
        raise ValueError("Expected probabilities array with shape [patients, 2]")
    if len(set(y.tolist())) != 2:
        raise ValueError("AUC requires both TLS classes in the evaluation partition")
    pred = np.argmax(p, axis=1)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p[:, 1])),
        "ppv": float(tp / (tp + fp)) if tp + fp else 0.0,
        "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
    }


@torch.no_grad()
def predict(model: GeneImageMultimodalModel, loader, device):
    model.eval()
    probabilities = []
    true_labels = []
    for genes, images, labels in loader:
        genes = genes.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(genes, images)
        probabilities.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        true_labels.append(labels.cpu().numpy())
    if not probabilities:
        raise ValueError("Evaluation loader was empty")
    return np.concatenate(true_labels), np.concatenate(probabilities)


def checkpoint_model_state(path: Path):
    checkpoint = safe_torch_load(path, "cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Expected dict with model_state_dict in {path}")
    return checkpoint
