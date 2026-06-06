"""Compatibility test on a real HuggingFace architecture.

Builds a tiny BERT *from config* (random weights — no network download) so CI
stays fast and offline, and confirms deploy-doctor handles a real, deep,
attention-based model end to end. Skipped if `transformers` isn't installed.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from deploy_doctor.doctor import diagnose  # noqa: E402
from deploy_doctor.inspect import inspect_model  # noqa: E402


def _tiny_bert():
    cfg = transformers.BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return transformers.BertModel(cfg).eval()


def test_real_bert_int8_is_flagged():
    model = _tiny_bert()
    qmodel = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    rep = inspect_model(qmodel)
    assert rep.quant_scheme == "dynamic-int8"
    # BERT has many Linear layers; all should be flagged CPU-locked.
    assert sum(1 for m in rep.modules if m.quantized) >= 10
    assert diagnose(qmodel, target_device="cuda", do_bench=False).verdict == "FAIL"


def test_real_bert_fp32_passes():
    assert diagnose(_tiny_bert(), target_device="cuda", do_bench=False).verdict == "PASS"
