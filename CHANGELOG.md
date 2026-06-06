# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [Unreleased]

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
