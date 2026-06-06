from torch import nn

from deploy_doctor.demo_models import build_fp32, quantize_int8_dynamic
from deploy_doctor.inspect import inspect_model, is_quantized_module


def test_fp32_model_reports_no_quantization():
    rep = inspect_model(build_fp32())
    assert rep.quant_scheme == "none"
    assert rep.has_quantized is False
    assert rep.devices == ["cpu"]
    assert any("float32" in m.dtypes for m in rep.modules)


def test_int8_model_detected_as_dynamic():
    rep = inspect_model(quantize_int8_dynamic(build_fp32()))
    assert rep.quant_scheme == "dynamic-int8"
    assert rep.has_quantized is True
    # The demo MLP has exactly three Linear layers.
    assert sum(1 for m in rep.modules if m.quantized) == 3
    # ...and we must NOT double-count the internal _packed_params submodule.
    assert len(rep.modules) == 3


def test_quantized_weights_live_on_cpu():
    rep = inspect_model(quantize_int8_dynamic(build_fp32()))
    for m in rep.modules:
        if m.quantized:
            assert m.devices == ["cpu"]
            assert "qint8" in m.dtypes


def test_is_quantized_module_predicate():
    q = quantize_int8_dynamic(build_fp32())
    leaves = [mod for mod in q.modules() if not list(mod.children())]
    assert any(is_quantized_module(mod) for mod in leaves)
    assert not is_quantized_module(nn.Linear(4, 4))


def test_paramless_modules_are_skipped():
    rep = inspect_model(nn.Sequential(nn.ReLU(), nn.Dropout()))
    # No parameters anywhere -> nothing to place -> no leaf rows.
    assert rep.modules == []
    assert rep.total_params == 0
