"""Synthetic encoder shape tests; the new feature exporter is tested separately."""
import torch

from cae_train.networks import CAEGen


def test_cae_encoder_shape_from_256_rgb_image():
    model = CAEGen().eval()
    with torch.inference_mode():
        content, style = model.encode(torch.zeros(1, 3, 256, 256))
    assert content.shape == (1, 256, 64, 64)
    assert style.shape == (1, 256, 1, 1)


def test_cae_style_dim_is_configurable():
    # The individual-specific code dimension is a constructor option.
    model = CAEGen(style_dim=8).eval()
    with torch.inference_mode():
        _, style = model.encode(torch.zeros(1, 3, 64, 64))
    assert style.shape == (1, 8, 1, 1)


def test_cae_default_256_style_code_can_decode():
    # Architecture smoke test only; no learned weights or real images.
    model = CAEGen(style_dim=256).eval()
    with torch.inference_mode():
        x = torch.zeros(1, 3, 64, 64)
        content, style = model.encode(x)
        decoded = model.decode(content, style)
    assert style.shape == (1, 256, 1, 1)
    assert decoded.shape == x.shape
