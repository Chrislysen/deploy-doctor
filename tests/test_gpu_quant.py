"""The anti-embarrassment tests.

bitsandbytes / torchao / GPTQ run int8/4-bit *on the GPU*. If deploy-doctor ever
flagged one of those as "CPU-locked", it would be instantly, visibly wrong to
anyone who ships quantized LLMs. These tests lock in that it never does — using
synthetic stand-ins, since those libraries aren't required to run the suite.
"""

from torch import nn

from deploy_doctor.doctor import diagnose
from deploy_doctor.fallback import detect_fallbacks
from deploy_doctor.inspect import inspect_model, is_eager_quantized, is_gpu_quantized


class _BnbLinear4bit(nn.Linear):
    """Stand-in for bitsandbytes.nn.Linear4bit (GPU-capable 4-bit)."""


_BnbLinear4bit.__module__ = "bitsandbytes.nn.modules"


def _bnb_model():
    return nn.Sequential(_BnbLinear4bit(16, 16), nn.ReLU(), _BnbLinear4bit(16, 16)).eval()


def test_bitsandbytes_is_not_cpu_locked():
    model = _bnb_model()
    leaves = [m for m in model.modules() if isinstance(m, _BnbLinear4bit)]
    assert all(is_gpu_quantized(m) for m in leaves)
    assert all(not is_eager_quantized(m) for m in leaves)

    findings = detect_fallbacks(model, target_device="cuda")
    codes = {f.code for f in findings}
    assert "int8_cpu_locked" not in codes  # the embarrassing false positive
    assert "gpu_quant_ok" in codes
    # No FAIL: a GPU-capable quantized model targeted at CUDA is correct.
    assert diagnose(model, target_device="cuda", do_bench=False).verdict == "PASS"


def test_gpu_quant_scheme_label():
    assert inspect_model(_bnb_model()).quant_scheme == "gpu-quant"


def test_quant_state_attr_is_recognized():
    m = nn.Linear(8, 8)
    m.quant_state = object()  # bitsandbytes attaches one of these
    assert is_gpu_quantized(m)


def test_plain_models_are_not_gpu_quant():
    assert not is_gpu_quantized(nn.Linear(8, 8))
    assert not is_gpu_quantized(nn.Linear(8, 8).half())


def test_eager_int8_still_flagged_regression():
    # The real footgun must still FAIL — we narrowed detection, didn't break it.
    from deploy_doctor.demo_models import build_fp32, quantize_int8_dynamic

    q = quantize_int8_dynamic(build_fp32())
    assert diagnose(q, target_device="cuda", do_bench=False).verdict == "FAIL"
