"""FastAPI entrypoint — /health /optimize /explain /research /risk-tags/apply /backtest /anova."""
from __future__ import annotations

from fastapi import FastAPI, Query

from api import __version__
from api.schemas import (
    AnovaRequest,
    AnovaResponse,
    BacktestResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    OptimizeRequest,
    OptimizeResponse,
    ResearchRequest,
    ResearchResponse,
    RiskTagsApplyRequest,
    RiskTagsApplyResponse,
)
from api.services.anova_svc import run_anova
from api.services.backtest_svc import run_backtest
from api.services.explain import run_explain
from api.services.optimize import run_optimize
from api.services.research import run_research
from api.services.risk_apply import run_risk_tags_apply

app = FastAPI(
    title="Robo-Advisor API",
    description=(
        "Educational skeleton wrapping RL / research modules. "
        "Not investment advice; backtests ≠ future returns. "
        "ANOVA endpoints use synthetic demo series for teaching only. "
        "RAG /research uses plan→retrieve→tag_risk→verify→summarize stub "
        "with in-memory store + citation placeholders. POST /risk-tags/apply demos causal panel → PortfolioEnv.portfolio_risk."
    ),
    version=__version__,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe used by Docker healthcheck and Streamlit Health tab."""
    return HealthResponse(
        version=__version__,
        educational=True,
        disclaimer=(
            "Educational demo only — not investment advice; backtests ≠ future returns."
        ),
    )


@app.post("/optimize", response_model=OptimizeResponse, tags=["portfolio"])
def optimize(body: OptimizeRequest) -> OptimizeResponse:
    return run_optimize(body)


@app.post("/explain", response_model=ExplainResponse, tags=["xai"])
def explain(body: ExplainRequest) -> ExplainResponse:
    return run_explain(body)


@app.post("/research", response_model=ResearchResponse, tags=["research"])
def research(body: ResearchRequest) -> ResearchResponse:
    return run_research(body)


@app.get("/backtest", response_model=BacktestResponse, tags=["eval"])
def backtest(
    n_days: int = Query(default=252, ge=20, le=2520),
    seed: int = Query(default=0),
    include_benchmark: bool = Query(default=True),
    benchmarks: str = Query(
        default="spy,kospi",
        description="Comma-separated logical benchmarks (synthetic if no CSV).",
    ),
) -> BacktestResponse:
    names = [b.strip() for b in benchmarks.split(",") if b.strip()] or ["spy", "kospi"]
    return run_backtest(
        n_days=n_days,
        seed=seed,
        include_benchmark=include_benchmark,
        benchmarks=names,
    )


@app.post(
    "/anova",
    response_model=AnovaResponse,
    tags=["eval"],
    summary="Educational ANOVA on synthetic series",
    description=(
        "Wraps rl.stats_tests.one_way_anova / two_way_anova on synthetic demo returns. "
        "Educational only — not investment advice; synthetic data ≠ live markets."
    ),
)
def anova(body: AnovaRequest) -> AnovaResponse:
    return run_anova(body)


@app.post(
    "/risk-tags/apply",
    response_model=RiskTagsApplyResponse,
    tags=["research"],
    summary="Apply risk tags → causal panel + PortfolioEnv.portfolio_risk demo",
    description=(
        "Educational wiring stub: accepts optional risk_tags (or mock / rag stub), "
        "builds a causal risk_score_panel via rl.risk_tags, and optionally steps a "
        "short PortfolioEnv on dummy data so portfolio_risk appears in observation. "
        "Not investment advice; no live LLM."
    ),
)
def risk_tags_apply(body: RiskTagsApplyRequest) -> RiskTagsApplyResponse:
    return run_risk_tags_apply(body)

