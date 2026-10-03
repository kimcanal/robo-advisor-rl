"""Risk-tag → PortfolioEnv.portfolio_risk wiring demo (educational stub).

No live LLM. Accepts optional tags or generates mock / RAG-graph tags, builds a
causal risk_score_panel via rl.risk_tags, and optionally steps a short
PortfolioEnv on dummy data so callers can see portfolio_risk in obs / info.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from api.schemas import (
    RiskTag,
    RiskTagsApplyRequest,
    RiskTagsApplyResponse,
)


def _pd_ts(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _tags_to_df(tags: list[RiskTag]) -> pd.DataFrame:
    rows = [
        {
            "ticker": t.ticker,
            "risk_score": float(t.risk_score),
            "tag": t.tag,
            "ts": t.ts,
        }
        for t in tags
    ]
    return pd.DataFrame(rows)


def _resolve_tags(req: RiskTagsApplyRequest) -> tuple[pd.DataFrame, str]:
    """Return (validated tags df, source label)."""
    from rl.risk_tags import mock_risk_tags, validate_risk_tags

    tickers = list(req.tickers) or ["SPY"]
    if req.risk_tags:
        df = validate_risk_tags(_tags_to_df(req.risk_tags))
        return df, "client_supplied"

    if req.source == "rag_graph":
        try:
            from rag.graph import run_research_graph

            state = run_research_graph(
                query=req.query,
                tickers=tickers,
                n_events_per_ticker=req.n_events_per_ticker,
                seed=req.seed,
                top_k=5,
            )
            return state.risk_tags_df, "rag_graph_stub"
        except Exception:  # noqa: BLE001 — fall back to mock
            pass

    df = mock_risk_tags(
        tickers,
        start=req.panel_start,
        end=req.panel_end,
        n_events_per_ticker=req.n_events_per_ticker,
        seed=req.seed,
    )
    return df, "mock_risk_tags"


def _panel_summary(panel: pd.DataFrame) -> dict[str, Any]:
    """Compact causal-panel summary (no full matrix dump)."""
    nonzero = (panel > 0).sum().sum()
    per_ticker = {
        str(c): {
            "mean": float(panel[c].mean()),
            "max": float(panel[c].max()),
            "nonzero_days": int((panel[c] > 0).sum()),
        }
        for c in panel.columns
    }
    return {
        "n_dates": int(len(panel)),
        "n_tickers": int(panel.shape[1]),
        "nonzero_cells": int(nonzero),
        "global_mean": float(panel.values.mean()) if panel.size else 0.0,
        "global_max": float(panel.values.max()) if panel.size else 0.0,
        "per_ticker": per_ticker,
        "date_start": str(panel.index.min().date()) if len(panel) else None,
        "date_end": str(panel.index.max().date()) if len(panel) else None,
    }


def _sample_env_obs(
    tickers: list[str],
    tags: pd.DataFrame,
    *,
    n_steps: int,
    window: int,
    seed: int,
) -> dict[str, Any]:
    """Short PortfolioEnv walk on dummy data; return sample portfolio_risk values."""
    from rl.env.portfolio_env import PortfolioEnv
    from rl.pipeline import prepare_env_inputs

    _, returns, rsi_df, macd_df = prepare_env_inputs(
        tickers,
        start="2019-01-01",
        end="2020-12-31",
        use_dummy=True,
        verbose=False,
    )
    env = PortfolioEnv(
        returns,
        rsi_df,
        macd_df,
        window=window,
        risk_tags=tags,
        reward_type="simple",
    )
    obs, _ = env.reset(seed=seed)
    samples: list[dict[str, Any]] = []
    # Equal-ish actions (zeros → softmax ≈ uniform)
    action = np.zeros(env.n_assets, dtype=np.float32)
    rng = np.random.default_rng(seed)
    for i in range(max(1, n_steps)):
        obs, _reward, terminated, truncated, info = env.step(action)
        samples.append(
            {
                "step": i,
                "portfolio_risk": float(info.get("portfolio_risk", obs[-1])),
                "obs_portfolio_risk": float(obs[-1]) if obs is not None and len(obs) else None,
                "effective_mdd_limit": float(info.get("effective_mdd_limit", float("nan"))),
            }
        )
        # mild action jitter so weights move a bit
        action = rng.normal(0.0, 0.1, size=env.n_assets).astype(np.float32)
        if terminated or truncated:
            break

    risks = [s["portfolio_risk"] for s in samples]
    return {
        "n_steps_run": len(samples),
        "obs_dim": int(env.observation_space.shape[0]),
        "n_risk_obs": int(PortfolioEnv.N_RISK_OBS),
        "sample_portfolio_risk": samples[: min(10, len(samples))],
        "portfolio_risk_mean": float(np.mean(risks)) if risks else 0.0,
        "portfolio_risk_max": float(np.max(risks)) if risks else 0.0,
        "saw_nonzero_risk": bool(any(r > 0 for r in risks)),
    }


def run_risk_tags_apply(req: RiskTagsApplyRequest) -> RiskTagsApplyResponse:
    t0 = time.perf_counter()
    from rl.risk_tags import risk_score_panel

    tickers = list(req.tickers) or ["SPY"]
    tags_df, source = _resolve_tags(req)
    dates = pd.bdate_range(req.panel_start, req.panel_end)
    panel = risk_score_panel(tags_df, tickers, dates)
    summary = _panel_summary(panel)

    env_demo: dict[str, Any] | None = None
    if req.run_env_demo:
        env_demo = _sample_env_obs(
            tickers,
            tags_df,
            n_steps=req.env_steps,
            window=req.env_window,
            seed=req.seed,
        )

    tag_models = [
        RiskTag(
            ticker=str(row["ticker"]),
            risk_score=float(row["risk_score"]),
            tag=str(row["tag"]),
            ts=str(_pd_ts(row["ts"])),
        )
        for _, row in tags_df.iterrows()
    ]
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return RiskTagsApplyResponse(
        tickers=tickers,
        source=source,
        risk_tags=tag_models,
        panel_summary=summary,
        env_demo=env_demo,
        stub=True,
        notes=(
            "Educational stub: causal risk_score_panel (no look-ahead) wired into "
            "PortfolioEnv observation axis portfolio_risk. Not investment advice; "
            "dummy/synthetic path only — no live LLM or broker."
        ),
        env_contract=(
            "risk_tags {ticker, risk_score[0,1], tag, ts} → "
            "rl.risk_tags.risk_score_panel → PortfolioEnv.portfolio_risk"
        ),
        latency_ms=latency_ms,
    )
