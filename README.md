# deploy-doctor

[![CI](https://github.com/Chrislysen/deploy-doctor/actions/workflows/ci.yml/badge.svg)](https://github.com/Chrislysen/deploy-doctor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%E2%80%933.14-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.13%2B-ee4c2c)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
![License](https://img.shields.io/badge/license-MIT-green)

**Does your "GPU" model actually run on the GPU?** A lot of the time, it doesn't —
and nothing tells you. `deploy-doctor` catches silent device-placement footguns
in PyTorch models before they cost you a 10× latency regression in production.

```bash
pip install deploy-doctor      # (coming soon) — or: pip install -e .
deploy-doctor demo
```

---

## The footgun

You want faster inference, so you quantize your model to int8:

```python
import torch
qmodel = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
qmodel.to("cuda")          # no error ✅
out = qmodel(x.to("cuda")) # no error ✅
```

No exception is raised. Looks like it's on the GPU. **It isn't.** PyTorch's
eager-mode int8 kernels (FBGEMM/QNNPACK) have **no CUDA backend**, so the compute
silently stays on the CPU — often *slower* than the fp32 model you started with,
on hardware you're paying for. This is the kind of thing that shows up as a
mysterious latency regression weeks later, not as a build failure.

`deploy-doctor` makes it loud:

```
=== int8 dynamic-quantized — the footgun (target: cuda) ===
deploy-doctor  ·  device-placement diagnosis
────────────────────────────────────────────────────────────
  quantization : dynamic-int8
  weights on   : cpu
  dtypes       : qint8×3
  params       : 2,097,152
  target device: cuda

  findings
  [✗] 3 int8-quantized module(s) cannot run on CUDA — they are locked to the CPU.
      PyTorch eager-mode int8 quantization uses FBGEMM/QNNPACK, which have
      no CUDA backend. Calling .to('cuda') will NOT move this compute to the
      GPU; it silently stays on the CPU. For GPU int8, export to TensorRT or
      use a CUDA-aware path instead.
      · net.0
      · net.2
      · net.4

────────────────────────────────────────────────────────────
  FAIL  — silent device-placement footgun detected
```

The `demo` reproduces this from scratch in a couple of seconds — **no GPU and no
downloads required**, because being CPU-locked is a *static* property of the
model. The same finding holds whether or not the machine running it has a GPU.

## Check your own model

Point it at any factory that returns an `nn.Module`:

```bash
deploy-doctor check mypkg.models:make_model --target cuda --input-shape 1,3,224,224
```

- `--target` — the device you *intend* to deploy on (default `cuda`).
- `--input-shape` — optional; enables a quick latency benchmark on the device
  the model actually runs on.
- `--fail-on {fail,warn,never}` — severity that makes the exit code non-zero
  (default `fail`). Use `--fail-on warn` to gate a pipeline on warnings too.
- `--json` — machine-readable output for CI.

It also works on real models. Point it at a small factory that loads (and, if
you like, quantizes) yours — e.g. a HuggingFace model:

```python
# myloaders.py
import torch
from transformers import AutoModel
def quantized_bert():
    m = AutoModel.from_pretrained("bert-base-uncased").eval()
    return torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
```
```bash
deploy-doctor check myloaders:quantized_bert --target cuda   # -> FAIL: CPU-locked
```

Exit code is non-zero on a `FAIL`, so you can wire it into a pipeline:

```yaml
- run: deploy-doctor check mypkg.models:make_model --target cuda --fail-on warn
```

Or use it as a library:

```python
from deploy_doctor import diagnose, render
result = diagnose(model, example_input=x, target_device="cuda")
print(render(result))
print(result.verdict)          # "PASS" | "WARN" | "FAIL"
print(result.to_dict())        # full structured report
```

## What it checks (v0.1)

| Check | Severity | What it catches |
|---|---|---|
| `int8_cpu_locked` | **FAIL** | int8-quantized modules targeted at CUDA — they can only run on CPU |
| `meta_unmaterialized` | **FAIL** | weights still on the `meta` device — the model was never actually loaded |
| `mixed_devices` | **WARN** | weights split across CPU/GPU — a `.to()` that missed a submodule or buffer |
| `train_mode_at_inference` | **WARN** | model left in `train()` mode — Dropout/BatchNorm silently corrupt inference |
| `fp16_on_cpu` | **WARN** | fp16 weights on a CPU target — upcasts/perf cliff, no speedup |
| `offloaded_weights` | **WARN** | accelerate CPU/disk offload — weights stream device↔device every forward pass |
| `float64_weights` | **WARN** | fp64 weights — silent ~32× throughput cliff on GPU |
| live confirmation | — | on a machine *with* CUDA, actually moves the model and proves where it lands |

Try it on a realistic architecture too — `deploy-doctor demo --arch transformer`
shows *partial* quantization (FFN Linears int8-locked, attention/LayerNorm fp32).

When CUDA **is** present, `deploy-doctor` also runs a live check: it moves the
model to the GPU and re-inspects, confirming empirically what the static analysis
predicted. When CUDA is absent it says so plainly and leans on the static finding
(which doesn't need a GPU to be true).

## Why trust the output

The analysis is **static and read-only** — it reasons about what the model *is*
(module types, packed-param blobs, tensor dtypes/devices), never guessing from a
single lucky run. The optional live check and the latency benchmark report the
**actual** device the output landed on, not the one you asked for. If a number
can't be verified on the current hardware, the tool says so instead of pretending.

## Scope & limitations

- Targets **PyTorch eager-mode quantization** (`quantize_dynamic` / static
  convert) — by far the most common path people ship. FX-graph / `torchao` and
  ONNX/TensorRT paths are on the roadmap.
- v0.1 focuses on **device-placement** footguns. Numerical-accuracy drift from
  quantization is a separate concern and out of scope here.
- "Will it run on GPU" is about kernel availability, not whether GPU is *faster*
  for your shape — though the benchmark gives you the real latency to decide.

## Background

This tool productizes an empirical finding from my research on constrained ML
deployment: PyTorch's int8 dynamic quantization silently relocates inference to
the CPU, which turned a "feasible" deployment infeasible in **39%** of cases in a
multi-GPU study. The research repo:
[Constrained-ML-Deployment](https://github.com/Chrislysen/Constrained-ML-Deployment).

## Develop

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

See [CHANGELOG.md](CHANGELOG.md) for what's new.

## License

MIT © Christian Lysenstøen
