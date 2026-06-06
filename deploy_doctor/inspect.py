"""Static inspection of a PyTorch model: device placement, dtypes, quantization.

Everything here is read-only and works without a GPU — it reasons about what a
model *is*, not what happens when you run it. That is deliberate: the most
damaging deployment footgun (an int8 model that can only ever run on CPU) is a
static property of the model and can be detected on any machine.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import torch
from torch import nn

# Quantized tensor dtypes (CPU-only in PyTorch eager-mode quantization).
QUANT_DTYPES = {torch.qint8, torch.quint8, torch.qint32}


def is_eager_quantized(module: nn.Module) -> bool:
    """True if ``module`` is a PyTorch *eager-mode* quantized module.

    These use the FBGEMM/QNNPACK backends and are **CPU/ARM-only** — the footgun
    this tool is built around. Detected by the packed-params blob or torch's own
    quantized namespace (covers ``torch.ao.nn.quantized`` and
    ``torch.ao.nn.intrinsic.quantized``). Crucially scoped to ``torch.*`` so it
    never catches a GPU-capable third-party library.
    """
    if hasattr(module, "_packed_params"):
        return True
    namespace = getattr(type(module), "__module__", "") or ""
    return namespace.startswith("torch.") and "quantized" in namespace


# Backwards-compatible alias used across the codebase and tests.
is_quantized_module = is_eager_quantized

_GPU_QUANT_MARKERS = ("bitsandbytes", "torchao", "auto_gptq", "gptq", "awq", "exllama")


def is_gpu_quantized(module: nn.Module) -> bool:
    """True if ``module`` is **GPU-capable** quantization — bitsandbytes
    (``Linear4bit``/``Linear8bitLt``), torchao, GPTQ, AWQ, etc.

    These run int8/4-bit *on the GPU* and must **never** be reported as
    CPU-locked. This is the check that keeps the tool from embarrassing itself in
    front of anyone who actually ships quantized LLMs.
    """
    if is_eager_quantized(module):
        return False
    namespace = (getattr(type(module), "__module__", "") or "").lower()
    if any(marker in namespace for marker in _GPU_QUANT_MARKERS):
        return True
    name = type(module).__name__.lower()
    if any(k in name for k in ("4bit", "8bit", "nf4", "int4")):
        return True
    if hasattr(module, "quant_state") or hasattr(module, "weight_bit_width"):
        return True
    # torchao stores an AffineQuantized tensor subclass as a plain Linear weight.
    w = getattr(module, "weight", None)
    if w is not None and not callable(w):
        wmod = (getattr(type(w), "__module__", "") or "").lower()
        wname = type(w).__name__.lower()
        if "torchao" in wmod or "affinequant" in wname or "quantized" in wname:
            return True
    return False


def _is_dynamic_quantized(module: nn.Module) -> bool:
    namespace = getattr(type(module), "__module__", "") or ""
    return "quantized" in namespace and "dynamic" in namespace


def _local_tensors(module: nn.Module):
    """Yield (name, tensor) for this module's *own* params and buffers."""
    for name, param in module.named_parameters(recurse=False):
        yield name, param
    for name, buf in module.named_buffers(recurse=False):
        yield name, buf


def _module_devices_dtypes(module: nn.Module):
    devices: set[str] = set()
    dtypes: set[str] = set()
    n_params = 0
    for _, tensor in _local_tensors(module):
        if tensor is None:
            continue
        devices.add(tensor.device.type)
        dtypes.add(str(tensor.dtype).replace("torch.", ""))
        n_params += tensor.numel()

    if is_quantized_module(module):
        # Packed int8 weights don't surface as normal params/buffers. Try the
        # quantized accessor; fall back to the known truth: FBGEMM is CPU-only.
        try:
            w = module.weight() if callable(getattr(module, "weight", None)) else None
            if w is not None:
                devices.add(w.device.type)
                dtypes.add(str(w.dtype).replace("torch.", ""))
                n_params += w.numel()
        except Exception:
            pass
        if not devices:
            devices.add("cpu")
        if not dtypes:
            dtypes.add("qint8")
    return sorted(devices), sorted(dtypes), n_params


@dataclasses.dataclass
class ModuleInfo:
    name: str
    type: str
    devices: list[str]
    dtypes: list[str]
    quantized: bool  # eager-mode, CPU-locked
    param_count: int
    gpu_quantized: bool = False  # bitsandbytes / torchao / GPTQ — GPU-capable


@dataclasses.dataclass
class ModelReport:
    modules: list[ModuleInfo]
    devices: list[str]
    dtypes: dict[str, int]
    quant_scheme: str
    total_params: int
    has_quantized: bool

    def to_dict(self) -> dict:
        return {
            "quant_scheme": self.quant_scheme,
            "devices": self.devices,
            "dtypes": self.dtypes,
            "total_params": self.total_params,
            "has_quantized": self.has_quantized,
            "modules": [dataclasses.asdict(m) for m in self.modules],
        }


def _classify_scheme(modules: list[ModuleInfo], dtype_counts: Counter) -> str:
    any_dynamic = any(m.quantized and "dynamic" in m.type.lower() for m in modules)
    any_quant = any(m.quantized for m in modules)
    any_gpu_quant = any(m.gpu_quantized for m in modules)
    quant_dtype = any(
        d in {"qint8", "quint8", "qint32"} for m in modules for d in m.dtypes
    )
    if any_dynamic:
        return "dynamic-int8"
    if any_quant or quant_dtype:
        return "static-int8"
    if any_gpu_quant:
        return "gpu-quant"
    float_dtypes = {d for d in dtype_counts if d.startswith("float") or d == "half"}
    if float_dtypes == {"float16"} or float_dtypes == {"half"}:
        return "fp16"
    if len(float_dtypes) > 1:
        return "mixed-precision"
    return "none"


def inspect_model(model: nn.Module) -> ModelReport:
    """Walk ``model`` and summarise device placement, dtype, and quantization."""
    leaf_infos: list[ModuleInfo] = []
    all_devices: set[str] = set()
    dtype_counts: Counter = Counter()
    total_params = 0
    quant_prefixes: list[str] = []

    for name, module in model.named_modules():
        # Skip anything nested *inside* a quantized module (e.g. the internal
        # _packed_params blob) — the quantized module is itself the leaf.
        if any(name.startswith(p + ".") for p in quant_prefixes):
            continue
        # Leaf modules only: a module with no children, OR a quantized module
        # (whose packed internals we don't want to descend into).
        children = list(module.children())
        if is_quantized_module(module):
            quant_prefixes.append(name)
        elif children:
            continue
        devices, dtypes, n = _module_devices_dtypes(module)
        if not devices and not dtypes and n == 0 and not is_quantized_module(module):
            continue  # paramless op (ReLU, Dropout, ...) — nothing to place

        quant = is_eager_quantized(module)
        gpu_quant = is_gpu_quantized(module)
        type_name = type(module).__module__ + "." + type(module).__name__
        if _is_dynamic_quantized(module):
            type_name += "  (dynamic)"
        leaf_infos.append(
            ModuleInfo(
                name=name or "<root>",
                type=type_name,
                devices=devices,
                dtypes=dtypes,
                quantized=quant,
                param_count=n,
                gpu_quantized=gpu_quant,
            )
        )
        all_devices.update(devices)
        for d in dtypes:
            dtype_counts[d] += 1
        total_params += n

    scheme = _classify_scheme(leaf_infos, dtype_counts)
    return ModelReport(
        modules=leaf_infos,
        devices=sorted(all_devices),
        dtypes=dict(dtype_counts),
        quant_scheme=scheme,
        total_params=total_params,
        has_quantized=any(m.quantized for m in leaf_infos),
    )
