"""Backtest metrics — synthetic portfolio path + SPY/KOSPI synthetic benchmarks."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from api.schemas import BacktestResponse
from rl.backtest import compute_metrics
from rl.benchmarks import synthetic_benchmark_returns


def _clean_metrics(raw: dict) -> dict:
    return {
        k: (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else float(v))
        for k, v in raw.items()
    }


def run_backtest(
    n_days: int = 252,
    seed: int = 0,
    include_benchmark: bool = True,
    benchmarks: list[str] | None = None,
) -> BacktestResponse:
    t0 = time.perf_counter()
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
        "Smoke backtest via rl.backtest.compute_metrics on synthetic portfolio returns. "
        "Benchmarks use rl.benchmarks.synthetic_benchmark_returns (SPY/KOSPI-shaped) "
        "so CI needs no network; swap for load_benchmark_returns + local CSV for report numbers. "
        f"latency_ms={latency_ms}."
    )
    return BacktestResponse(
        metrics=metrics,
        n_days=n_days,
        method="synthetic_equal_path+synth_benchmarks",
        notes=notes,
        benchmark_metrics=benchmark_metrics,
        latency_ms=latency_ms,
    )
