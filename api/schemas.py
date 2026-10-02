"""Pydantic request/response models for the public API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "robo-advisor-api"
    version: str = "0.1.4"
    educational: bool = True
    disclaimer: str = (
        "Educational demo only — not investment advice; backtests ≠ future returns."
    )


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
    latency_ms: float | None = None


class ExplainRequest(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "AGG"])
    asset_index: int = Field(default=0, ge=0, description="Which asset weight to explain")
    top_k: int = Field(default=5, ge=1, le=20)
    model_path: str | None = Field(
        default=None,
        description="Optional SB3 zip under rl/outputs/models; ignored if missing.",
    )
    artifact_path: str | None = Field(
        default=None,
        description="Optional precomputed SHAP JSON artifact path.",
    )


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class ExplainResponse(BaseModel):
    asset: str
    asset_index: int
    top_features: list[FeatureContribution]
    summary: str
    stub: bool = True
    mode: str = Field(
        default="stub_pseudo_shap",
        description="stub_pseudo_shap | artifact_json | shap_kernel_attempted",
    )
    latency_ms: float | None = None


class ResearchRequest(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "TLT"])
    query: str = Field(default="market risk outlook", min_length=1)
    n_events_per_ticker: int = Field(default=3, ge=1, le=10)
    top_k: int = Field(default=5, ge=1, le=20, description="Retrieval top-k from stub store")


class RiskTag(BaseModel):
    ticker: str
    risk_score: float = Field(ge=0.0, le=1.0)
    tag: str
    ts: str


class Citation(BaseModel):
    doc_id: str
    title: str
    source: str
    ticker: str
    score: float = 0.0
    snippet: str = ""
    quote: str = ""


class NodeLatency(BaseModel):
    """Per-node wall time from the stub graph (educational timings, not SLOs)."""

    node: str
    latency_ms: float


class ResearchResponse(BaseModel):
    query: str
    risk_tags: list[RiskTag]
    report_excerpt: str
    stub: bool = True
    plan: list[str] = Field(default_factory=list)
    node_trace: list[str] = Field(
        default_factory=list,
        description="Ordered stub nodes: plan → retrieve → tag_risk → verify → summarize",
    )
    node_latencies_ms: list[NodeLatency] = Field(
        default_factory=list,
        description="Per-node stub timings matching node_trace order",
    )
    citations: list[Citation] = Field(default_factory=list)
    verify_ok: bool = False
    verify_notes: list[str] = Field(default_factory=list)
    env_contract: str = Field(
        default=(
            "risk_tags schema {ticker, risk_score[0,1], tag, ts} → "
            "PortfolioEnv observation axis portfolio_risk (holdings-weighted mean)"
        )
    )
    latency_ms: float | None = None


class BacktestRequest(BaseModel):
    """Optional body for POST-style backtest; GET uses query params instead."""
    n_days: int = Field(default=252, ge=20, le=2520)
    seed: int = 0
    include_benchmark: bool = True
    benchmarks: list[str] = Field(
        default_factory=lambda: ["spy", "kospi"],
        description="Logical benchmark names (synthetic when no local CSV).",
    )


class BacktestResponse(BaseModel):
    metrics: dict[str, Any]
    n_days: int
    method: str
    notes: str = ""
    benchmark_metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    latency_ms: float | None = None
    latency_notes: str = (
        "Smoke path is synthetic (no network). Real Walk-Forward / yfinance "
        "benchmarks add I/O latency; prefer local CSV under rl/data/raw/."
    )


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
    latency_ms: float | None = None


class RiskTagsApplyRequest(BaseModel):
    """Wire risk tags into a causal panel + optional short PortfolioEnv demo."""

    tickers: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "TLT"])
    risk_tags: list[RiskTag] | None = Field(
        default=None,
        description="Optional client-supplied tags; if omitted, generated via source.",
    )
    source: str = Field(
        default="mock",
        description="When risk_tags omitted: 'mock' or 'rag_graph' (stub, no live LLM).",
    )
    query: str = Field(default="market risk outlook", min_length=1)
    n_events_per_ticker: int = Field(default=3, ge=1, le=10)
    panel_start: str = Field(default="2019-01-01")
    panel_end: str = Field(default="2020-12-31")
    run_env_demo: bool = Field(
        default=True,
        description="If true, step a short PortfolioEnv on dummy data and sample portfolio_risk.",
    )
    env_steps: int = Field(default=8, ge=1, le=40)
    env_window: int = Field(default=20, ge=5, le=60)
    seed: int = Field(default=0)


class RiskTagsApplyResponse(BaseModel):
    tickers: list[str]
    source: str
    risk_tags: list[RiskTag]
    panel_summary: dict[str, Any]
    env_demo: dict[str, Any] | None = None
    stub: bool = True
    notes: str = ""
    env_contract: str = (
        "risk_tags {ticker, risk_score[0,1], tag, ts} → "
        "rl.risk_tags.risk_score_panel → PortfolioEnv.portfolio_risk"
    )
    latency_ms: float | None = None

