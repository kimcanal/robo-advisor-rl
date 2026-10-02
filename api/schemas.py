"""Pydantic request/response models for the public API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "robo-advisor-api"
    version: str = "0.1.0"


class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(
        default_factory=lambda: [
            "SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "TLT", "HYG", "GLD", "VNQ"
        ],
        description="Asset universe (symbols).",
    )
    method: str = Field(
        default="mvo",
        description="Optimization method: 'mvo' (Markowitz) or 'equal' (1/N).",
    )
    lookback_days: int = Field(default=252, ge=30, le=1260)
    objective: str = Field(default="max_sharpe", description="mvo objective")


class OptimizeResponse(BaseModel):
    method: str
    tickers: list[str]
    weights: dict[str, float]
    notes: str = ""


class ExplainRequest(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "AGG"])
    asset_index: int = Field(default=0, ge=0, description="Which asset weight to explain")
    top_k: int = Field(default=5, ge=1, le=20)


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class ExplainResponse(BaseModel):
    asset: str
    asset_index: int
    top_features: list[FeatureContribution]
    summary: str
    stub: bool = True


class ResearchRequest(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "TLT"])
    query: str = Field(default="market risk outlook", min_length=1)
    n_events_per_ticker: int = Field(default=3, ge=1, le=10)


class RiskTag(BaseModel):
    ticker: str
    risk_score: float = Field(ge=0.0, le=1.0)
    tag: str
    ts: str


class ResearchResponse(BaseModel):
    query: str
    risk_tags: list[RiskTag]
    report_excerpt: str
    stub: bool = True


class BacktestRequest(BaseModel):
    """Optional body for POST-style backtest; GET uses query params instead."""
    n_days: int = Field(default=252, ge=20, le=2520)
    seed: int = 0
    include_benchmark: bool = True


class BacktestResponse(BaseModel):
    metrics: dict[str, Any]
    n_days: int
    method: str
    notes: str = ""


class AnovaRequest(BaseModel):
    """Educational ANOVA on synthetic demo series (no live market data)."""

    mode: str = Field(
        default="one_way",
        description="ANOVA mode: 'one_way' (DRL vs MVO vs equal) or 'two_way' (strategy × regime).",
    )
    n_obs: int = Field(default=120, ge=30, le=5000, description="Observations per group / total budget")
    seed: int = Field(default=0, description="RNG seed for reproducible synthetic series")


class AnovaResponse(BaseModel):
    mode: str
    groups: list[str]
    n_obs: int
    result: dict[str, Any]
    notes: str = (
        "Educational statistical demo wrapping rl.stats_tests. "
        "Not investment advice; synthetic data ≠ live markets."
    )
    stub: bool = True

