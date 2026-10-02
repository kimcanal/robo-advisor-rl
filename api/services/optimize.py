"""Portfolio weight optimization — wraps rl.mvo when possible."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.schemas import OptimizeRequest, OptimizeResponse


def _equal_weights(tickers: list[str]) -> dict[str, float]:
    n = len(tickers)
    w = 1.0 / n if n else 0.0
    return {t: round(w, 6) for t in tickers}


def _synthetic_returns(tickers: list[str], lookback_days: int, seed: int = 42) -> pd.DataFrame:
    """GBM-ish synthetic log returns for offline / no-data environments."""
    rng = np.random.default_rng(seed)
    n = len(tickers)
    factor = rng.normal(0.0003, 0.01, size=lookback_days)
    idio = rng.normal(0.0, 0.008, size=(lookback_days, n))
    rets = factor[:, None] * 0.5 + idio
    idx = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=lookback_days)
    return pd.DataFrame(rets, index=idx, columns=tickers)


def run_optimize(req: OptimizeRequest) -> OptimizeResponse:
    tickers = list(req.tickers)
    if not tickers:
        return OptimizeResponse(method=req.method, tickers=[], weights={}, notes="empty universe")

    if req.method == "equal":
        return OptimizeResponse(
            method="equal",
            tickers=tickers,
            weights=_equal_weights(tickers),
            notes="1/N baseline (no model load).",
        )

    returns: pd.DataFrame | None = None
    notes = ""
    try:
        from rl.data.loader import load_price_data
        from rl.features import log_returns

        prices = load_price_data(tickers, use_dummy=True, verbose=False)
        log_ret = log_returns(prices).dropna()
        if len(log_ret) >= min(30, req.lookback_days):
            returns = log_ret.tail(req.lookback_days)
            notes = "MVO on rl.data.loader prices (CSV or dummy fallback)."
    except Exception as exc:  # pragma: no cover - defensive path
        notes = f"loader unavailable ({type(exc).__name__}); using synthetic returns."

    if returns is None or returns.empty:
        returns = _synthetic_returns(tickers, req.lookback_days)
        notes = notes or "MVO on synthetic returns (no price panel)."

    from rl.mvo import optimize_weights

    w = optimize_weights(returns, objective=req.objective)
    weights = {t: round(float(wi), 6) for t, wi in zip(tickers, w)}
    s = sum(weights.values()) or 1.0
    weights = {t: round(v / s, 6) for t, v in weights.items()}
    return OptimizeResponse(method="mvo", tickers=tickers, weights=weights, notes=notes)
