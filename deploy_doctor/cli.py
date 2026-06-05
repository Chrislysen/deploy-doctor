"""Command-line interface for deploy-doctor.

    deploy-doctor demo                 # reproduce the int8 CPU-fallback footgun
    deploy-doctor check my.module:make_model --target cuda
    deploy-doctor check my.module:make_model --json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

import torch

from . import __version__
from .demo_models import build_demo, example_input
from .doctor import diagnose, render


def _load_model(spec: str):
    """Load a model from ``package.module:callable`` (or ``:attribute``)."""
    if ":" not in spec:
        raise SystemExit(
            f"error: --model must be 'package.module:factory', got '{spec}'"
        )
    mod_name, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(mod_name)
    except Exception as exc:
        raise SystemExit(f"error: could not import '{mod_name}': {exc}")
    obj = getattr(module, attr, None)
    if obj is None:
        raise SystemExit(f"error: '{attr}' not found in '{mod_name}'")
    model = obj() if callable(obj) else obj
    if not isinstance(model, torch.nn.Module):
        raise SystemExit(f"error: '{spec}' did not produce a torch.nn.Module")
    return model


def _parse_shape(shape: str | None):
    if not shape:
        return None
    try:
        dims = [int(s) for s in shape.replace("x", ",").split(",") if s.strip()]
    except ValueError:
        raise SystemExit(f"error: bad --input-shape '{shape}' (use e.g. 1,512)")
    return torch.randn(*dims)


def _emit(result, target, as_json: bool):
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(render(result, target_device=target, color=sys.stdout.isatty()))


def cmd_demo(args: argparse.Namespace) -> int:
    fp32, int8, x = build_demo()
    target = args.target

    if not args.json:
        print("\n=== fp32 baseline (target: %s) ===" % target)
    res_fp32 = diagnose(fp32, x, target_device=target, do_bench=True)
    _emit(res_fp32, target, args.json)

    if not args.json:
        print("\n=== int8 dynamic-quantized — the footgun (target: %s) ===" % target)
    res_int8 = diagnose(int8, x, target_device=target, do_bench=True)
    _emit(res_int8, target, args.json)
    return 1 if res_int8.verdict == "FAIL" else 0


def cmd_check(args: argparse.Namespace) -> int:
    model = _load_model(args.model)
    x = _parse_shape(args.input_shape)
    result = diagnose(model, x, target_device=args.target, do_bench=x is not None)
    _emit(result, args.target, args.json)
    return {"FAIL": 1, "WARN": 0, "PASS": 0}[result.verdict]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deploy-doctor",
        description="Catch silent ML deployment footguns (e.g. int8 models that "
        "silently run on CPU instead of the GPU you asked for).",
    )
    p.add_argument("--version", action="version", version=f"deploy-doctor {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="reproduce the int8 silent-CPU-fallback footgun")
    d.add_argument("--target", default="cuda", help="device you intend to deploy on")
    d.add_argument("--json", action="store_true", help="machine-readable output")
    d.set_defaults(func=cmd_demo)

    c = sub.add_parser("check", help="diagnose your own model")
    c.add_argument("model", help="model factory as 'package.module:callable'")
    c.add_argument("--target", default="cuda", help="device you intend to deploy on")
    c.add_argument(
        "--input-shape",
        default=None,
        help="example input shape for latency benchmark, e.g. '1,512'",
    )
    c.add_argument("--json", action="store_true", help="machine-readable output")
    c.set_defaults(func=cmd_check)
    return p


def main(argv: list[str] | None = None) -> int:
    # Box-drawing/verdict glyphs are UTF-8; older Windows consoles default to
    # cp1252 and would mojibake them. Best-effort upgrade, never fatal.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
