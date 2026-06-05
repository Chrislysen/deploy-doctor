"""Built-in demo: reproduce the int8 silent-CPU-fallback footgun from scratch.

No downloads, no GPU required. We build a small fp32 MLP, dynamically quantize
it to int8 (the exact thing people do for "faster inference"), and hand both
back so the doctor can show that the quantized one is CPU-locked.
"""

from __future__ import annotations

import torch
from torch import nn


class TinyMLP(nn.Module):
    """A small Linear-stack — the part of a model int8 dynamic quant targets."""

    def __init__(self, in_dim: int = 512, hidden: int = 1024, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_fp32(in_dim: int = 512) -> nn.Module:
    model = TinyMLP(in_dim=in_dim)
    model.eval()
    return model


def quantize_int8_dynamic(model: nn.Module) -> nn.Module:
    """Apply the most common 'make it faster' move: dynamic int8 quantization."""
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )


def example_input(in_dim: int = 512, batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, in_dim)


def build_demo():
    """Return ``(fp32_model, int8_model, example_input)`` for the demo."""
    fp32 = build_fp32()
    int8 = quantize_int8_dynamic(build_fp32())
    return fp32, int8, example_input()
