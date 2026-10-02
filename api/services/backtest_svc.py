"""Backtest metrics — wraps rl.backtest.compute_metrics on a simple path."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.schemas import BacktestResponse
from rl.backtest import compute_metrics


def run_backtest(n_days: int = 252, seed: int = 0, include_benchmark: bool = True) -> BacktestResponse:
    rng = np.random.default_rng(seed)
    # Equal-ish portfolio log-return path for smoke metrics (no heavy train).
    port = pd.Series(rng.normal(0.0004, 0.01, size=n_days))
    port.index = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n_days)
    bench = None
    if include_benchmark:
        bench = pd.Series(rng.normal(0.0003, 0.009, size=n_days), index=port.index)

    raw = compute_metrics(port, benchmark=bench, risk_free=0.0)
    metrics = {
        k: (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else float(v))
        for k, v in raw.items()
    }
    return BacktestResponse(
        metrics=metrics,
        n_days=n_days,
        method="synthetic_equal_path",
        notes=(
            "Smoke backtest via rl.backtest.compute_metrics on synthetic returns. "
            "Replace with Walk-Forward / trained policy paths for report numbers."
        ),
    )
