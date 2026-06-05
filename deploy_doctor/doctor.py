"""Orchestration: tie inspect + fallback + (optional) live check + bench together,
and render a human-readable verdict.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from .bench import benchmark
from .fallback import Finding, detect_fallbacks, verify_live
from .inspect import ModelReport, inspect_model

VERDICT_RANK = {"FAIL": 3, "WARN": 2, "PASS": 1}


@dataclasses.dataclass
class DiagnoseResult:
    verdict: str  # "PASS" | "WARN" | "FAIL"
    report: ModelReport
    findings: list[Finding]
    live: Finding | None
    bench: dict | None

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "report": self.report.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "live": self.live.to_dict() if self.live else None,
            "bench": self.bench,
        }


def _verdict(findings: list[Finding], live: Finding | None) -> str:
    sev = [f.severity for f in findings]
    if live:
        sev.append(live.severity)
    if "fail" in sev:
        return "FAIL"
    if "warn" in sev:
        return "WARN"
    return "PASS"


def diagnose(
    model: nn.Module,
    example_input=None,
    target_device: str = "cuda",
    do_bench: bool = True,
) -> DiagnoseResult:
    report = inspect_model(model)
    findings = detect_fallbacks(model, target_device=target_device, report=report)
    live = verify_live(model, target_device=target_device)
    bench = None
    if do_bench and example_input is not None:
        try:
            bench = benchmark(model, example_input)
        except Exception as exc:  # benchmarking must never break the diagnosis
            bench = {"error": f"{type(exc).__name__}: {exc}"}
    verdict = _verdict(findings, live)
    return DiagnoseResult(verdict, report, findings, live, bench)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

_ICON = {"fail": "✗", "warn": "!", "info": "i", "ok": "✓"}
_VERDICT_LINE = {
    "FAIL": "FAIL  — silent device-placement footgun detected",
    "WARN": "WARN  — review the findings below",
    "PASS": "PASS  — no device-placement footguns detected",
}


def render(result: DiagnoseResult, target_device: str = "cuda", color: bool = True) -> str:
    def c(code: str, text: str) -> str:
        if not color:
            return text
        return f"\033[{code}m{text}\033[0m"

    r = result.report
    lines: list[str] = []
    lines.append(c("1", "deploy-doctor") + "  ·  device-placement diagnosis")
    lines.append("─" * 60)

    # Model summary
    lines.append(f"  quantization : {r.quant_scheme}")
    lines.append(f"  weights on   : {', '.join(r.devices) or 'n/a'}")
    lines.append(f"  dtypes       : {', '.join(f'{k}×{v}' for k, v in r.dtypes.items()) or 'n/a'}")
    lines.append(f"  params       : {r.total_params:,}")
    lines.append(f"  target device: {target_device}")
    lines.append("")

    # Findings
    lines.append("  findings")
    for f in result.findings:
        col = {"fail": "31", "warn": "33", "info": "36", "ok": "32"}.get(f.severity, "0")
        lines.append("  " + c(col, f"[{_ICON.get(f.severity, '?')}] {f.message}"))
        if f.detail:
            for chunk in _wrap(f.detail, 70):
                lines.append("      " + c("2", chunk))
        for m in f.modules[:6]:
            lines.append("      " + c("2", f"· {m}"))
        if len(f.modules) > 6:
            lines.append("      " + c("2", f"· … and {len(f.modules) - 6} more"))

    # Live confirmation
    if result.live:
        lines.append("")
        col = {"fail": "31", "ok": "32"}.get(result.live.severity, "0")
        lines.append("  live check (CUDA present on this machine)")
        lines.append("  " + c(col, f"[{_ICON.get(result.live.severity, '?')}] {result.live.message}"))
        if result.live.detail:
            lines.append("      " + c("2", result.live.detail))
    else:
        lines.append("")
        lines.append("  " + c("2", "live check : skipped (no CUDA device on this machine — "))
        lines.append("  " + c("2", "             the finding above is static and holds regardless)"))

    # Bench
    if result.bench and "error" not in result.bench:
        b = result.bench
        lines.append("")
        lines.append("  latency")
        lines.append(
            "      "
            + c("2", f"ran on {b['device_output']}  ·  {b['mean_ms']} ms mean  ·  "
                     f"p95 {b['p95_ms']} ms  ·  {b['fps']} it/s")
        )

    lines.append("")
    lines.append("─" * 60)
    vcol = {"FAIL": "31;1", "WARN": "33;1", "PASS": "32;1"}.get(result.verdict, "0")
    lines.append("  " + c(vcol, _VERDICT_LINE.get(result.verdict, result.verdict)))
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out
