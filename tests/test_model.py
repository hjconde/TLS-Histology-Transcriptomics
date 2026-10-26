"""Synthetic tests: these create no biomedical data or study labels."""
import torch
from torch import nn

from tls_multimodal.model import (EnhancedImageEncoder, CrossModalAttentionFusion,
                                  GeneImageMultimodalModel, ContrastiveLoss,
                                  MemoryEfficientSelfAttention, FocalLoss)


class FakeGeneEncoder(nn.Module):
    """Unit-test stub ONLY; NEVER used as a fallback in train_5fold.py."""
    def __init__(self, dim=200):
        super().__init__()
        self.tokens = nn.Embedding(13, dim)

    def forward(self, x, return_encodings=False):
        result = self.tokens(x)
        return (result, None) if return_encodings else result


def test_original_parameter_names_maintained():
    attn = MemoryEfficientSelfAttention(200, 4)
    keys = list(attn.state_dict())
    assert "qkv.weight" in keys
    assert "out_proj.weight" in keys


def test_multimodal_forward_and_gradients():
    torch.manual_seed(123)
    model = GeneImageMultimodalModel(
        FakeGeneEncoder(), EnhancedImageEncoder(),
        CrossModalAttentionFusion(), num_classes=2,
    )
    model.train()
    gene_ids = torch.randint(0, 12, (3, 7))
    image = torch.randn(3, 1, 256)
    labels = torch.tensor([1, 0, 1])
    logits, contrast = model(gene_ids, image, labels=labels, contrast_weight=0.2)
    assert logits.shape == (3, 2)
    assert contrast.ndim == 0
    assert torch.isfinite(logits).all()
    total = FocalLoss(alpha=torch.tensor([1.0, 0.5]))(logits, labels) + contrast
    total.backward()
    assert model.fusion.fusion_mlp[0].weight.grad is not None
    model.eval()
    with torch.no_grad():
        assert model(gene_ids, image).shape == (3, 2)


def test_bidirectional_contrastive_loss():
    crit = ContrastiveLoss(temperature=0.1)
    a = torch.eye(3)
    assert 0 <= crit(a, a) < crit(a, a[[2, 1, 0]])
