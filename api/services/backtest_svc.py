"""Backtest metrics.

source=walk_forward (default when docs/results/wf_summary.json exists):
    the committed Walk-Forward 12-metric table + truncation flags + target check.
source=synthetic: smoke path on synthetic series (CI / no data). Clearly labelled.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from api.schemas import BacktestResponse
from api.services.artifacts import read_json, results_dir
from rl.backtest import compute_metrics
from rl.benchmarks import synthetic_benchmark_returns


def _clean_metrics(raw: dict) -> dict:
    return {
        k: (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else float(v))
        for k, v in raw.items()
    }


def _walk_forward(t0: float) -> BacktestResponse | None:
    summary = read_json(results_dir() / "wf_summary.json")
    if not summary:
        return None
    csv_path = results_dir() / summary["source"].split("/")[-1]
    rows = []
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        trunc = {(r["window"], r["strategy"], r["seed"]) for r in summary.get("truncated_rows", [])}
        for r in df.to_dict(orient="records"):
            r = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}
            r["truncated"] = (r["window"], r["strategy"], r["seed"]) in trunc
            rows.append(r)
    drl = [r for r in rows if str(r["strategy"]).startswith("drl_")]
    mean_metrics = {}
    if drl:
        keys = [k for k in drl[0] if k not in {"window", "strategy", "seed", "truncated"}]
        for k in keys:
            vals = [r[k] for r in drl if isinstance(r.get(k), (int, float)) and r[k] is not None]
            mean_metrics[k] = float(np.mean(vals)) if vals else None
    kospi = "present" if any(r["strategy"] == "kospi" for r in rows) else (
        "SKIP — KOSPI download failed in every window (no KOSPI rows in results)"
    )
    return BacktestResponse(
        metrics=mean_metrics,
        n_days=int(max((w.get("ew_days") or 0) for w in summary["windows"])) if summary["windows"] else 0,
        method="walk_forward_results_csv",
        source="walk_forward",
        windows=summary["windows"],
        rows=rows,
        truncated_rows=summary.get("truncated_rows", []),
        targets={k: v for k, v in summary.get("targets", {}).items() if k != "rows"},
        kospi_status=kospi,
        notes=(
            "metrics = mean over all DRL rows (all rewards/seeds/windows). "
            + summary.get("truncation_note", "")
        ),
        latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        latency_notes="Reads committed results; no training or download at request time.",
    )


def run_backtest(
    n_days: int = 252,
    seed: int = 0,
    include_benchmark: bool = True,
    benchmarks: list[str] | None = None,
    source: str = "auto",
) -> BacktestResponse:
    t0 = time.perf_counter()
    if source in ("auto", "walk_forward"):
        wf = _walk_forward(t0)
        if wf is not None:
            return wf
        if source == "walk_forward":
            return BacktestResponse(metrics={}, n_days=0, method="unavailable", source="walk_forward",
                                    notes="docs/results/wf_summary.json missing — run rl.analysis.wf_analysis")
    rng = np.random.default_rng(seed)
    # Equal-ish portfolio log-return path for smoke metrics (no heavy train).
    port = pd.Series(rng.normal(0.0004, 0.01, size=n_days))
    port.index = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n_days)

    names = list(benchmarks) if benchmarks else ["spy", "kospi"]
    primary_bench = None
    benchmark_metrics: dict[str, dict] = {}

    if include_benchmark:
        # Primary bench for alpha/beta on the portfolio metrics = first name (spy).
        for i, name in enumerate(names):
            # Distinct seeds so spy/kospi differ but stay reproducible.
            series = synthetic_benchmark_returns(
                port.index,
                mu=0.0003 if name.lower() == "spy" else 0.00025,
                sigma=0.009 if name.lower() == "spy" else 0.011,
                seed=seed + 10 + i,
                name=name.lower(),
            )
            if primary_bench is None:
                primary_bench = series
            # Standalone metrics for each benchmark (no nested benchmark).
            benchmark_metrics[name.lower()] = _clean_metrics(
                compute_metrics(series, benchmark=None, risk_free=0.0)
            )

    raw = compute_metrics(port, benchmark=primary_bench, risk_free=0.0)
    metrics = _clean_metrics(raw)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    notes = (
        "SYNTHETIC — not market data. Smoke backtest via rl.backtest.compute_metrics on synthetic portfolio returns. "
        "Benchmarks use rl.benchmarks.synthetic_benchmark_returns (SPY/KOSPI-shaped) "
        "so CI needs no network; swap for load_benchmark_returns + local CSV for report numbers. "
        f"latency_ms={latency_ms}."
    )
    return BacktestResponse(
        metrics=metrics,
        n_days=n_days,
        source="synthetic",
        method="synthetic_equal_path+synth_benchmarks",
        notes=notes,
        benchmark_metrics=benchmark_metrics,
        latency_ms=latency_ms,
    )
