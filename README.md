# deploy-doctor

[![CI](https://github.com/Chrislysen/deploy-doctor/actions/workflows/ci.yml/badge.svg)](https://github.com/Chrislysen/deploy-doctor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%E2%80%933.14-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.13%2B-ee4c2c)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
![License](https://img.shields.io/badge/license-MIT-green)

**Does your "GPU" model actually run on the GPU?** Sometimes it doesn't, and
PyTorch won't tell you. `deploy-doctor` is a small CLI that catches silent
device-placement footguns in PyTorch models — and it works without a GPU.

```bash
pip install git+https://github.com/Chrislysen/deploy-doctor
deploy-doctor demo
```

## The main footgun

You quantize a model to int8 for "faster inference":

```python
qmodel = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
qmodel.to("cuda")   # no error — but the weights don't actually move
```

PyTorch eager-mode int8 (FBGEMM/QNNPACK) has **no CUDA backend**. The `.to("cuda")`
silently does nothing, and at inference you get either a hard `NotImplementedError`
(if the input is on the GPU) or the compute quietly running on the CPU — so the
speedup never happens. You find out in staging, not at build. `deploy-doctor`
finds it in CI, on a CPU runner, because being CPU-locked is a *static* property
of the model.

> int8/4-bit done with **bitsandbytes, torchao, or GPTQ/AWQ** *does* run on the
> GPU — deploy-doctor recognises those and reports them as fine, not a footgun.

## Usage

```bash
deploy-doctor demo                                   # reproduce the footgun (no GPU needed)
deploy-doctor demo --arch transformer                # on an attention model (partial quant)
deploy-doctor check mypkg.models:make_model --target cuda
```

`check` takes `--target` (intended device), `--input-shape 1,512` (adds a latency
benchmark), `--fail-on {fail,warn,never}` (CI exit code), and `--json`. The exit
code is non-zero on `FAIL`, so it drops straight into a pipeline.

As a library:

```python
from deploy_doctor import diagnose
result = diagnose(model, example_input=x, target_device="cuda")
result.verdict     # "PASS" | "WARN" | "FAIL"
result.to_dict()   # structured report
```

## What it checks

| Check | Severity | What it catches |
|---|---|---|
| `int8_cpu_locked` | **FAIL** | eager-mode int8 modules targeted at CUDA — they can only run on CPU |
| `meta_unmaterialized` | **FAIL** | weights still on the `meta` device — the model was never loaded |
| `mixed_devices` | **WARN** | weights split across CPU/GPU — a `.to()` that missed a submodule |
| `train_mode_at_inference` | **WARN** | model left in `train()` mode — Dropout/BatchNorm corrupt inference |
| `fp16_on_cpu` | **WARN** | fp16 weights on a CPU target — upcasts/perf cliff, no speedup |
| `offloaded_weights` | **WARN** | accelerate CPU/disk offload — weights stream device↔device per pass |
| `float64_weights` | **WARN** | fp64 weights — large throughput cliff on GPU |
| `gpu_quant_ok` | **OK** | bitsandbytes / torchao / GPTQ — GPU-capable, explicitly *not* flagged |

The analysis is static and read-only. On a machine with CUDA it also runs a live
check (moves the model and reports where it actually lands); without one it says
so and leans on the static finding, which doesn't need a GPU to be true.

## Scope

- Targets PyTorch **eager-mode** quantization (`quantize_dynamic` / static
  convert) — the common case. FX-graph, torchao, and ONNX/TensorRT paths are
  not covered yet.
- Focuses on **device placement**, not numerical-accuracy drift.
- "Runs on GPU" means kernel availability, not whether the GPU is faster for
  your shape (the benchmark gives you the real latency to decide).

## Background

Productizes a finding from my
[Constrained-ML-Deployment](https://github.com/Chrislysen/Constrained-ML-Deployment)
research: PyTorch int8 dynamic quantization silently moves inference to the CPU,
which flipped ~39% of "feasible" deployments to infeasible in a multi-GPU study.

## Develop

```bash
pip install -e ".[dev]" && pytest -q && ruff check .
```

MIT © Christian Lysenstøen
