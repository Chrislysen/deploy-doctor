"""Measure real inference latency on the device the model *actually* runs on."""

from __future__ import annotations

import time

import torch
from torch import nn


def _model_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def _to_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(e, device) for e in x)
    return x


def _as_args(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


def _output_device(out) -> str:
    if isinstance(out, torch.Tensor):
        return out.device.type
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
        return out[0].device.type
    return "unknown"


def benchmark(
    model: nn.Module,
    example_input,
    runs: int = 30,
    warmup: int = 5,
) -> dict:
    """Time ``runs`` forward passes and report where the output actually landed.

    The ``device_output`` field is the honest one: if you asked for CUDA but it
    says ``cpu``, the compute silently fell back.
    """
    model.eval()
    device = _model_device(model)
    x = _to_device(example_input, device)
    is_cuda = device.type == "cuda"

    out = None
    with torch.no_grad():
        for _ in range(max(1, warmup)):
            out = model(*_as_args(x))
        if is_cuda:
            torch.cuda.synchronize()

        times: list[float] = []
        with torch.no_grad():
            for _ in range(max(1, runs)):
                t0 = time.perf_counter()
                out = model(*_as_args(x))
                if is_cuda:
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    mean = sum(times) / len(times)
    p95_idx = max(0, int(len(times) * 0.95) - 1)
    return {
        "device_model": device.type,
        "device_output": _output_device(out),
        "mean_ms": round(mean, 3),
        "p50_ms": round(times[len(times) // 2], 3),
        "p95_ms": round(times[p95_idx], 3),
        "fps": round(1000.0 / mean, 1) if mean > 0 else None,
        "runs": len(times),
    }
