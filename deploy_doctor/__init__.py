"""deploy-doctor — catch silent ML deployment footguns before they hit production.

The headline check: a model you *think* runs on the GPU may silently run on the
CPU. PyTorch's eager-mode int8 quantization (FBGEMM/QNNPACK) has **no CUDA
backend**, so a dynamically-quantized model stays on the CPU no matter how many
times you call ``.to("cuda")`` — often a large, invisible latency regression.

``deploy-doctor`` inspects a model, reports where its compute *actually* runs,
and flags these silent fallbacks with a clear verdict.
"""

from .bench import benchmark
from .doctor import DiagnoseResult, diagnose, render
from .fallback import Finding, detect_fallbacks
from .inspect import ModelReport, ModuleInfo, inspect_model

__version__ = "0.1.0"

__all__ = [
    "inspect_model",
    "ModelReport",
    "ModuleInfo",
    "detect_fallbacks",
    "Finding",
    "benchmark",
    "diagnose",
    "DiagnoseResult",
    "render",
    "__version__",
]
