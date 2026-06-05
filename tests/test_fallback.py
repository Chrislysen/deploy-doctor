import torch
from torch import nn

from deploy_doctor.demo_models import build_fp32, example_input, quantize_int8_dynamic
from deploy_doctor.doctor import diagnose
from deploy_doctor.fallback import detect_fallbacks


def _codes(findings):
    return {f.code for f in findings}


def test_int8_targeting_cuda_fails():
    findings = detect_fallbacks(quantize_int8_dynamic(build_fp32()), target_device="cuda")
    assert "int8_cpu_locked" in _codes(findings)
    assert any(f.severity == "fail" for f in findings)


def test_int8_targeting_cpu_is_ok():
    findings = detect_fallbacks(quantize_int8_dynamic(build_fp32()), target_device="cpu")
    assert "int8_cpu_locked" not in _codes(findings)
    assert all(f.severity != "fail" for f in findings)


def test_fp32_targeting_cuda_is_clean():
    findings = detect_fallbacks(build_fp32(), target_device="cuda")
    assert _codes(findings) == {"clean"}
    assert all(f.severity == "ok" for f in findings)


def test_mixed_device_placement_warns():
    # Force a mixed-device report without needing a GPU by hand-building it.
    from deploy_doctor.inspect import ModelReport, ModuleInfo

    rep = ModelReport(
        modules=[
            ModuleInfo("a", "Linear", ["cpu"], ["float32"], False, 10),
            ModuleInfo("b", "Linear", ["cuda"], ["float32"], False, 10),
        ],
        devices=["cpu", "cuda"],
        dtypes={"float32": 2},
        quant_scheme="none",
        total_params=20,
        has_quantized=False,
    )
    findings = detect_fallbacks(build_fp32(), target_device="cuda", report=rep)
    assert "mixed_devices" in _codes(findings)


def test_diagnose_verdicts():
    x = example_input()
    assert diagnose(build_fp32(), x, target_device="cuda").verdict == "PASS"
    assert diagnose(quantize_int8_dynamic(build_fp32()), x, target_device="cuda").verdict == "FAIL"
    assert diagnose(quantize_int8_dynamic(build_fp32()), x, target_device="cpu").verdict in ("PASS", "WARN")


def test_benchmark_runs_on_cpu():
    res = diagnose(build_fp32(), example_input(), target_device="cpu").bench
    assert res["device_output"] == "cpu"
    assert res["mean_ms"] > 0
    assert res["runs"] > 0
