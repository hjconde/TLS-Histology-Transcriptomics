"""TLS histology–transcriptomics multimodal architecture.

Provides image encoding, bidirectional cross-modal attention, patient-paired
contrastive loss and attention pooling. The bundled PerformerLM provides
the gene encoder. PyTorch SDPA implements memory-efficient self-attention.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.detach().clone())
        else:
            self.alpha = None
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt).pow(self.gamma) * ce).mean()


class ContrastiveLoss(nn.Module):
    """Symmetric patient-matched gene--image InfoNCE."""
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Contrastive temperature must be > 0")
        self.temperature = temperature

    def forward(self, gene_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        if gene_feat.shape[0] < 2:
            return gene_feat.new_zeros(())
        gene_feat = F.normalize(gene_feat, dim=-1)
        img_feat = F.normalize(img_feat, dim=-1)
        sim = gene_feat @ img_feat.T / self.temperature
        labels = torch.arange(sim.shape[0], device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0


class MemoryEfficientSelfAttention(nn.Module):
    """Multi-head self-attention with qkv/out_proj linear layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 batch_first: bool = True):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key=None, value=None, need_weights=False):
        x = query if self.batch_first else query.transpose(0, 1)
        batch, seq_len, channels = x.shape
        qkv = (self.qkv(x)
               .reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        # No explicit N x N attention matrix is materialized here when a
        # suitable optimized SDPA kernel is available.
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(batch, seq_len, channels)
        out = self.out_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, None  # Attention weights are not requested by this model.


class EnhancedImageEncoder(nn.Module):
    def __init__(self, img_dim=256, hidden_dim=256, out_dim=200, img_seq_len=1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(img_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.self_attn = MemoryEfficientSelfAttention(hidden_dim, num_heads=4,
                                                     dropout=0.1, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        attended, _ = self.self_attn(x, x, x)
        return self.norm(self.output_proj(x + attended))


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, gene_dim=200, img_dim=200, out_dim=200,
                 num_heads=4, dropout_rate=0.1, skip=True):
        super().__init__()
        self.skip = skip
        self.out_dim = out_dim
        self.gene_proj = nn.Linear(gene_dim, out_dim) if gene_dim != out_dim else nn.Identity()
        self.img_proj = nn.Linear(img_dim, out_dim) if img_dim != out_dim else nn.Identity()
        self.cross_attn_g2i = nn.MultiheadAttention(out_dim, num_heads,
                         dropout=dropout_rate, batch_first=True)
        self.cross_attn_i2g = nn.MultiheadAttention(out_dim, num_heads,
                         dropout=dropout_rate, batch_first=True)
        self.self_attn_gene = MemoryEfficientSelfAttention(out_dim, num_heads,
                                            dropout=dropout_rate, batch_first=True)
        self.self_attn_img = MemoryEfficientSelfAttention(out_dim, num_heads,
                                            dropout=dropout_rate, batch_first=True)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_dim * 4, out_dim), nn.LayerNorm(out_dim),
            nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.norm3 = nn.LayerNorm(out_dim)
        self.norm4 = nn.LayerNorm(out_dim)

    def forward(self, gene_feat, img_feat):
        gene_feat_proj = self.norm1(self.gene_proj(gene_feat))
        img_feat_proj = self.norm2(self.img_proj(img_feat))
        gene_self, _ = self.self_attn_gene(gene_feat_proj, gene_feat_proj, gene_feat_proj)
        img_self, _ = self.self_attn_img(img_feat_proj, img_feat_proj, img_feat_proj)
        gene_self = self.norm3(gene_self + gene_feat_proj)
        img_self = self.norm4(img_self + img_feat_proj)
        gene_global = gene_self.mean(dim=1)
        img_global = img_self.squeeze(1)
        # Explicitly disable attention-weight output: avoids a large allocation.
        g2i_attn, _ = self.cross_attn_g2i(
            gene_feat_proj, img_global.unsqueeze(1), img_global.unsqueeze(1),
            need_weights=False,
        )
        i2g_attn, _ = self.cross_attn_i2g(
            img_global.unsqueeze(1), gene_feat_proj, gene_feat_proj,
            need_weights=False,
        )
        seq_len = gene_feat_proj.shape[1]
        concatenated = torch.cat([
            gene_feat_proj,
            img_feat_proj.expand(-1, seq_len, -1),
            g2i_attn,
            i2g_attn.expand(-1, seq_len, -1),
        ], dim=-1)
        fused = self.fusion_mlp(concatenated)
        if self.skip:
            fused = fused + gene_feat_proj
        return fused, gene_global, img_global


class AdaptiveClassifierHead(nn.Module):
    def __init__(self, in_dim=200, hidden_dim=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.attention_pool = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4), nn.Tanh(),
            nn.Linear(in_dim // 4, 1), nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        attention = self.attention_pool(x)
        return self.classifier(torch.sum(x * attention, dim=1))


class GeneImageMultimodalModel(nn.Module):
    def __init__(self, gene_model: nn.Module, img_encoder: nn.Module,
                 fusion_module: nn.Module, num_classes=2, gene_out_dim=200):
        super().__init__()
        self.gene_encoder = gene_model
        self.img_encoder = img_encoder
        self.fusion = fusion_module
        self.classifier = AdaptiveClassifierHead(in_dim=gene_out_dim,
                                                 hidden_dim=64, num_classes=num_classes)
        self.norm = nn.LayerNorm(gene_out_dim)
        self.contrastive_loss = ContrastiveLoss(temperature=0.1)

    def forward(self, gene_data, img_data, labels=None,
                return_encodings=False, contrast_weight=0.1):
        try:
            encoding_result = self.gene_encoder(gene_data, return_encodings=True)
        except TypeError as exc:
            raise RuntimeError(
                "The installed performer_pytorch is not compatible with the "
                "original gaBERT forward(return_encodings=True) interface. "
                "Use the matching version of your research code."
            ) from exc
        gene_feat = encoding_result[0] if isinstance(encoding_result, tuple) else encoding_result
        if gene_feat.ndim != 3 or gene_feat.shape[-1] != self.norm.normalized_shape[0]:
            raise ValueError(f"Unexpected gaBERT output shape: {tuple(gene_feat.shape)}")
        gene_feat = self.norm(gene_feat)
        image_feat = self.img_encoder(img_data)
        fused, gene_global, image_global = self.fusion(gene_feat, image_feat)
        if return_encodings:
            return fused
        logits = self.classifier(fused)
        if self.training and labels is not None:
            contrast = self.contrastive_loss(gene_global, image_global) * contrast_weight
            return logits, contrast
        return logits
