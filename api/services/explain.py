"""Decision explanation stub — full SHAP needs a trained SB3 model on disk.

Avoid importing rl.shap_explain at module load (it pulls shap/matplotlib).
Feature-name layout mirrors PortfolioEnv observation order.
"""
from __future__ import annotations

import numpy as np

from api.schemas import ExplainRequest, ExplainResponse, FeatureContribution


def _feature_names(tickers: list[str], window: int = 30, *, include_risk: bool = True) -> list[str]:
    names: list[str] = []
    for k in range(window, 0, -1):
        for t in tickers:
            names.append(f"ret_t-{k}_{t}")
    names += [f"weight_{t}" for t in tickers]
    names += [f"rsi_{t}" for t in tickers]
    names += [f"macd_{t}" for t in tickers]
    if include_risk:
        names.append("portfolio_risk")
    return names


def run_explain(req: ExplainRequest) -> ExplainResponse:
    tickers = list(req.tickers) or ["SPY"]
    idx = min(req.asset_index, len(tickers) - 1)
    asset = tickers[idx]
    names = _feature_names(tickers, window=30, include_risk=True)

    # Deterministic pseudo-SHAP ranking without loading a PPO zip (keeps CI light).
    rng = np.random.default_rng(abs(hash(asset)) % (2**32))
    scores = rng.normal(0, 1, size=len(names))
    order = np.argsort(-np.abs(scores))[: req.top_k]
    top = [
        FeatureContribution(feature=names[i], contribution=round(float(scores[i]), 4))
        for i in order
    ]
    return ExplainResponse(
        asset=asset,
        asset_index=idx,
        top_features=top,
        summary=(
            f"Stub explanation for {asset} weight (index {idx}). "
            "Wire rl.shap_explain.explain_decision once models exist under "
            "rl/outputs/models/*.zip."
        ),
        stub=True,
    )
