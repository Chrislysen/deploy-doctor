from torch import nn

from deploy_doctor.demo_models import build_demo, build_fp32
from deploy_doctor.doctor import diagnose
from deploy_doctor.fallback import detect_fallbacks
from deploy_doctor.inspect import inspect_model


def _codes(findings):
    return {f.code for f in findings}


def test_train_mode_at_inference_warns():
    model = build_fp32().train()  # left in train mode
    findings = detect_fallbacks(model, target_device="cpu")
    assert "train_mode_at_inference" in _codes(findings)
    assert diagnose(model, target_device="cpu", do_bench=False).verdict == "WARN"


def test_eval_mode_does_not_warn():
    findings = detect_fallbacks(build_fp32().eval(), target_device="cpu")
    assert "train_mode_at_inference" not in _codes(findings)


def test_meta_device_is_flagged():
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)).to("meta")
    rep = inspect_model(model)
    assert "meta" in rep.devices
    findings = detect_fallbacks(model, target_device="cuda", report=rep)
    assert "meta_unmaterialized" in _codes(findings)
    assert any(f.severity == "fail" for f in findings)


def test_transformer_demo_partial_quantization():
    fp32, int8, x = build_demo("transformer")
    rep = inspect_model(int8)
    # Realistic partial quantization: some Linears int8, attention/norms fp32.
    assert rep.quant_scheme == "dynamic-int8"
    assert any(m.quantized for m in rep.modules)
    assert any(not m.quantized for m in rep.modules)
    assert diagnose(int8, x, target_device="cuda").verdict == "FAIL"


def test_transformer_int8_actually_runs():
    _, int8, x = build_demo("transformer")
    res = diagnose(int8, x, target_device="cpu", do_bench=True)
    assert res.bench is not None and "error" not in res.bench
    assert res.bench["mean_ms"] > 0


def test_fail_on_exit_code_mapping():
    from deploy_doctor.cli import _exit_code

    assert _exit_code("PASS", "fail") == 0
    assert _exit_code("WARN", "fail") == 0  # warnings don't fail by default
    assert _exit_code("WARN", "warn") == 1  # ...but can be opted into
    assert _exit_code("FAIL", "fail") == 1
    assert _exit_code("FAIL", "never") == 0  # never fail the build
