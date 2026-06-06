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


class _EncoderBlock(nn.Module):
    """One pre-norm transformer block: self-attention + feed-forward.

    Written out by hand (rather than nn.TransformerEncoderLayer) so the forward
    pass doesn't hit PyTorch's fused fast-path, which itself breaks on
    dynamically-quantized Linears — a separate footgun we don't want clouding
    the demo. Dynamic int8 quant reaches the two FFN Linears; MultiheadAttention
    keeps its packed in-projection (and its NonDynamicallyQuantizable out-proj)
    in fp32, so this shows realistic *partial* quantization.
    """

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x


class TinyTransformer(nn.Module):
    """A small but real Transformer encoder — attention + feed-forward."""

    def __init__(self, d_model: int = 256, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_EncoderBlock(d_model, nhead) for _ in range(layers)]
        )
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def build_transformer(d_model: int = 256) -> nn.Module:
    model = TinyTransformer(d_model=d_model)
    model.eval()
    return model


def transformer_input(d_model: int = 256, seq: int = 16, batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, seq, d_model)


def build_demo(arch: str = "mlp"):
    """Return ``(fp32_model, int8_model, example_input)`` for the demo.

    ``arch`` is ``"mlp"`` (default) or ``"transformer"``.
    """
    if arch == "transformer":
        return build_transformer(), quantize_int8_dynamic(build_transformer()), transformer_input()
    return build_fp32(), quantize_int8_dynamic(build_fp32()), example_input()
