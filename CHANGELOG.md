# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [Unreleased]

### Changed
- **Precise quant detection.** Distinguish PyTorch *eager-mode* int8
  (FBGEMM/QNNPACK, CPU-locked) from *GPU-capable* quantization (bitsandbytes,
  torchao, GPTQ/AWQ). GPU-capable quant is now reported as fine (`gpu_quant_ok`)
  and never false-flagged as CPU-locked.
- **Honest messaging.** The `int8_cpu_locked` finding now spells out the two real
  failure modes (hard `NotImplementedError` on CUDA input vs. silent CPU
  execution) instead of implying a single "silent slowdown".

### Added
- `deploy-doctor demo --arch transformer` — reproduces the footgun on a real
  attention encoder, showing *partial* int8 quantization (FFN Linears locked,
  attention/LayerNorm stay fp32).
- New checks: `meta_unmaterialized` (FAIL), `train_mode_at_inference` (WARN),
  `offloaded_weights` (WARN, accelerate CPU/disk offload), `float64_weights`
  (WARN, GPU throughput cliff).
- `deploy-doctor check --fail-on {fail,warn,never}` to control the CI exit code.
- Latency-contrast summary in the demo (fp32 vs int8 on the current device).
- Compatibility test on a real HuggingFace BERT built from config (offline).
- GitHub Actions CI (Python 3.11/3.12) and ruff linting.

## [0.1.0]

### Added
- Static device-placement diagnosis: `inspect_model`, `detect_fallbacks`.
- Flagship `int8_cpu_locked` check plus `mixed_devices` and `fp16_on_cpu`.
- Optional live CUDA confirmation and a latency benchmark.
- CLI (`demo`, `check`), JSON output, and a library API (`diagnose`/`render`).
- MIT license, `pyproject.toml`, and an initial test suite.
