"""Detect silent device-placement footguns.

A *finding* is something that will bite you in production but raises no error at
build time. The flagship one: you quantize a model to int8 for "faster GPU
inference", call ``.to("cuda")``, see no error — and every forward pass quietly
runs on the CPU because PyTorch's int8 kernels (FBGEMM/QNNPACK) have no CUDA
backend.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from .inspect import ModelReport, inspect_model

SEVERITY_ORDER = {"fail": 3, "warn": 2, "info": 1, "ok": 0}


@dataclasses.dataclass
class Finding:
    severity: str  # "fail" | "warn" | "info" | "ok"
    code: str
    message: str
    detail: str = ""
    modules: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def detect_fallbacks(
    model: nn.Module,
    target_device: str = "cuda",
    report: ModelReport | None = None,
) -> list[Finding]:
    """Return findings for ``model`` assuming you intend to run on ``target_device``.

    Pure static analysis — never runs the model, never needs a GPU.
    """
    report = report or inspect_model(model)
    target = torch.device(target_device).type
    findings: list[Finding] = []

    quant_modules = [m.name for m in report.modules if m.quantized]

    # 1. The headline footgun: int8 quantized model targeted at a CUDA device.
    if quant_modules and target == "cuda":
        findings.append(
            Finding(
                severity="fail",
                code="int8_cpu_locked",
                message=(
                    f"{len(quant_modules)} int8-quantized module(s) cannot run on "
                    f"CUDA — they are locked to the CPU."
                ),
                detail=(
                    "PyTorch eager-mode int8 quantization uses FBGEMM/QNNPACK, "
                    "which have no CUDA backend. Calling .to('cuda') will NOT move "
                    "this compute to the GPU; it silently stays on the CPU. For GPU "
                    "int8, export to TensorRT or use a CUDA-aware path instead."
                ),
                modules=quant_modules,
            )
        )

    # 2. Quantized model with no GPU intent — fine, but say so explicitly.
    if quant_modules and target == "cpu":
        findings.append(
            Finding(
                severity="ok",
                code="int8_cpu_ok",
                message=f"{len(quant_modules)} int8 module(s) will run on CPU (as intended).",
                modules=quant_modules,
            )
        )

    # 3. Mixed device placement — some weights on CPU, some on GPU.
    real_devices = [d for d in report.devices if d in ("cpu", "cuda", "mps")]
    if len(set(real_devices)) > 1:
        split = {
            d: [m.name for m in report.modules if d in m.devices]
            for d in set(real_devices)
        }
        findings.append(
            Finding(
                severity="warn",
                code="mixed_devices",
                message=f"Model weights are split across devices: {sorted(set(real_devices))}.",
                detail=(
                    "A .to(device) call probably missed a submodule or a registered "
                    "buffer. Cross-device forward passes either error at runtime or "
                    "trigger silent host<->device copies every step."
                ),
                modules=[f"{d}: {', '.join(names)}" for d, names in split.items()],
            )
        )

    # 4. Unmaterialized 'meta' weights — the model was never actually loaded.
    meta_modules = [m.name for m in report.modules if "meta" in m.devices]
    if meta_modules:
        findings.append(
            Finding(
                severity="fail",
                code="meta_unmaterialized",
                message=(
                    f"{len(meta_modules)} module(s) still on the 'meta' device "
                    f"— weights are not loaded."
                ),
                detail=(
                    "Meta tensors carry shape/dtype but no data (a common state "
                    "after `init_empty_weights()` or a deferred load). Running this "
                    "as-is produces garbage or errors; load real weights with "
                    "load_state_dict(..., assign=True) or to_empty() first."
                ),
                modules=meta_modules,
            )
        )

    # 5. Model left in train() mode at inference — silent correctness footgun.
    if model.training:
        findings.append(
            Finding(
                severity="warn",
                code="train_mode_at_inference",
                message=(
                    "Model is in train() mode — Dropout and BatchNorm "
                    "will misbehave at inference."
                ),
                detail=(
                    "In train() mode Dropout randomly zeros activations and "
                    "BatchNorm uses batch statistics instead of running stats, so "
                    "outputs become non-deterministic and batch-size dependent. "
                    "Call model.eval() before serving."
                ),
            )
        )

    # 6. fp16 weights but CPU is the target — CPU fp16 is a perf cliff.
    has_fp16 = any("float16" in m.dtypes or "half" in m.dtypes for m in report.modules)
    if has_fp16 and target == "cpu":
        findings.append(
            Finding(
                severity="warn",
                code="fp16_on_cpu",
                message="fp16 weights targeted at the CPU.",
                detail=(
                    "Most CPU kernels have poor or no native fp16 support, so ops "
                    "upcast to fp32 (extra copies, no speedup) or fall back to slow "
                    "paths. fp16 pays off on GPU, not CPU."
                ),
            )
        )

    # 7. accelerate CPU/disk offloading — weights move device every forward pass.
    offloaded = [
        name for name, m in model.named_modules() if hasattr(m, "_hf_hook")
    ]
    if offloaded:
        findings.append(
            Finding(
                severity="warn",
                code="offloaded_weights",
                message=f"{len(offloaded)} module(s) use accelerate offload hooks.",
                detail=(
                    "Weights are streamed from CPU/disk to the GPU on each forward "
                    "pass (device_map='auto' / disk offload). It runs, but the "
                    "per-step host<->device copies can dominate latency. Confirm "
                    "this is intended for your latency budget."
                ),
                modules=offloaded[:8],
            )
        )

    # 8. float64 weights — a silent ~32x throughput cliff on GPU.
    if any("float64" in m.dtypes or "double" in m.dtypes for m in report.modules):
        findings.append(
            Finding(
                severity="warn",
                code="float64_weights",
                message="float64 (double) weights detected.",
                detail=(
                    "Consumer/most datacenter GPUs run fp64 at a small fraction of "
                    "fp32 throughput (often ~1/32). This is usually an accidental "
                    ".double() or a numpy float64 that crept in. Cast to float32."
                ),
            )
        )

    if not findings:
        findings.append(
            Finding(
                severity="ok",
                code="clean",
                message=f"No device-placement footguns detected for target '{target}'.",
            )
        )

    findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 0), reverse=True)
    return findings


def verify_live(model: nn.Module, target_device: str = "cuda") -> Finding | None:
    """If a real CUDA device exists, *prove* the fallback by trying to move there.

    Returns ``None`` when no CUDA device is available (nothing can be proven
    live on this machine — the static finding stands on its own).
    """
    if torch.device(target_device).type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        moved = model.to(target_device)
    except Exception as exc:  # quantized .to('cuda') can raise outright
        return Finding(
            severity="fail",
            code="int8_cpu_locked_live",
            message="Confirmed live: moving the model to CUDA raised an error.",
            detail=f"{type(exc).__name__}: {exc}",
        )
    rep = inspect_model(moved)
    still_cpu = [m.name for m in rep.modules if m.quantized and "cpu" in m.devices]
    if still_cpu:
        return Finding(
            severity="fail",
            code="int8_cpu_locked_live",
            message="Confirmed live: after .to('cuda'), quantized weights are still on CPU.",
            modules=still_cpu,
        )
    return Finding(
        severity="ok",
        code="moved_to_cuda",
        message="Model moved to CUDA cleanly; no quantized CPU-locked modules remained.",
    )
