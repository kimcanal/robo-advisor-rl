"""Portfolio weights — trained DRL policy (seed ensemble), MVO, or 1/N."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fastapi import HTTPException

from api.schemas import OptimizeRequest, OptimizeResponse
from api.services.artifacts import drl_server


def _equal_weights(tickers: list[str]) -> dict[str, float]:
    n = len(tickers)
    return {t: round(1.0 / n, 6) for t in tickers} if n else {}


def _prices_from_request(req: OptimizeRequest) -> pd.DataFrame | None:
    if not req.prices:
        return None
    lengths = {len(v) for v in req.prices.values()}
    if len(lengths) != 1:
        raise HTTPException(422, "all price lists must have the same length")
    n = lengths.pop()
    idx = (
        pd.to_datetime(req.dates)
        if req.dates
        else pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    )
    if len(idx) != n:
        raise HTTPException(422, "len(dates) must equal len(prices[ticker])")
    return pd.DataFrame(req.prices, index=idx).sort_index()


def _run_drl(req: OptimizeRequest, t0: float) -> OptimizeResponse:
    server = drl_server()
    if not server.available:
        raise HTTPException(
            503,
            detail={
                "error": "no trained DRL policy",
                "status": server.status(),
                "hint": "run the Colab artifact cell, then commit artifacts/drl (see docs/rl_serving.md)",
            },
        )
    from rl.serving import load_local_prices

    tickers = server.runs[0].tickers
    prices = _prices_from_request(req)
    source_hint = "prices"
    if prices is None:
        prices = load_local_prices(tickers)
        source_hint = "local_csv" if prices is not None else "snapshot"
    cur = None
    if req.current_weights:
        cur = np.array([req.current_weights.get(t, 0.0) for t in tickers], dtype=float)
        if cur.sum() <= 0:
            raise HTTPException(422, "current_weights must sum to > 0")
        cur = cur / cur.sum()
    try:
        out = server.predict(prices=prices, current_weights=cur)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    data_source = source_hint if out["data_source"] == "prices" else "snapshot"
    st = server.status()
    return OptimizeResponse(
        method="drl",
        tickers=tickers,
        weights={t: round(w, 6) for t, w in out["weights"].items()},
        per_seed={k: {t: round(w, 6) for t, w in v.items()} for k, v in out["per_seed"].items()},
        as_of=out["as_of"],
        data_source=data_source,
        stub=False,
        model={k: st.get(k) for k in ("reward_type", "reward_kwargs", "seeds", "train", "test", "timesteps")}
        | {"run_tags": out["run_tags"]},
        notes=(
            f"PPO seed ensemble ({len(out['run_tags'])} seeds), reward={out['reward_type']}. "
            f"Decision for the next trading day after {out['as_of']} (data: {data_source}). "
            "Educational only — not investment advice."
        ),
        latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
    )


def run_optimize(req: OptimizeRequest) -> OptimizeResponse:
    t0 = time.perf_counter()
    method = (req.method or "mvo").lower()
    if method == "drl":
        return _run_drl(req, t0)

    tickers = list(req.tickers)
    if not tickers:
        return OptimizeResponse(method=method, tickers=[], weights={}, notes="empty universe",
                                data_source="none", stub=True)

    if method == "equal":
        return OptimizeResponse(
            method="equal", tickers=tickers, weights=_equal_weights(tickers),
            notes="1/N baseline (no model load).", data_source="none", stub=False,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        )

    from rl.features import log_returns
    from rl.mvo import optimize_weights
    from rl.serving import load_local_prices

    prices = _prices_from_request(req)
    data_source = "prices"
    if prices is None:
        prices = load_local_prices(tickers, tail=req.lookback_days + 1)
        data_source = "local_csv"
    if prices is None:
        from rl.data.loader import load_price_data

        prices = load_price_data(tickers, use_dummy=True, verbose=False)
        data_source = "dummy"
    returns = log_returns(prices[tickers]).dropna().tail(req.lookback_days)
    w = optimize_weights(returns, objective=req.objective)
    weights = {t: float(wi) for t, wi in zip(tickers, w)}
    s = sum(weights.values()) or 1.0
    weights = {t: round(v / s, 6) for t, v in weights.items()}
    note = {
        "prices": "MVO on request prices.",
        "local_csv": "MVO on rl/data/raw CSVs.",
        "dummy": "MVO on DUMMY prices (no CSV for these tickers) — illustration only.",
    }[data_source]
    return OptimizeResponse(
        method="mvo", tickers=tickers, weights=weights, notes=note,
        data_source=data_source, stub=data_source == "dummy",
        as_of=str(returns.index[-1].date()) if len(returns) else None,
        latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
    )
