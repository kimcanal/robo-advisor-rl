"""FastAPI entrypoint.

Spec endpoints: /health /optimize /explain /research /backtest
Extra (dashboard, read-only): /anova/results /training/curves /portfolio/history /artifacts/file
Demos: /anova (synthetic) /risk-tags/apply (stub)
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from api import __version__
from api.schemas import (
    AnovaRequest,
    ArtifactPayload,
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
from api.services.artifacts import artifact_status, safe_file
from api.services.backtest_svc import run_backtest
from api.services.explain import run_explain
from api.services.optimize import run_optimize
from api.services.research import run_research
from api.services.results import anova_results, portfolio_history, training_curves
from api.services.risk_apply import run_risk_tags_apply

OPENAPI_TAGS = [
    {"name": "ops", "description": "Liveness / grader smoke (health, educational disclaimer)."},
    {"name": "portfolio", "description": "Portfolio weights: trained PPO policy (drl), MVO, 1/N; test-year history."},
    {"name": "xai", "description": "SHAP of the trained policy (computed offline, served here)."},
    {
        "name": "research",
        "description": (
            "RAG LangGraph-style stub (plan→retrieve→tag_risk→verify→summarize) "
            "and risk-tag → PortfolioEnv.portfolio_risk wiring. Always stub: true; no live LLM."
        ),
    },
    {"name": "eval", "description": "Walk-forward 12 metrics, ANOVA results, learning / lambda curves."},
]

# Canonical public paths (also returned by GET /health for graders).
PUBLIC_ENDPOINTS = [
    "/health",
    "/optimize",
    "/explain",
    "/research",
    "/backtest",
    "/anova",
    "/risk-tags/apply",
    "/anova/results",
    "/training/curves",
    "/portfolio/history",
    "/artifacts/file",
]

app = FastAPI(
    title="Robo-Advisor API",
    description=(
        "Educational skeleton wrapping RL / research modules. "
        "Not investment advice; backtests ≠ future returns. "
        "ANOVA endpoints use synthetic demo series for teaching only. "
        "RAG /research uses plan→retrieve→tag_risk→verify→summarize stub "
        "with in-memory store + citation placeholders (RAG_TOP_K / RAG_STUB_FORCE / RAG_COLLECTION). "
        "POST /risk-tags/apply demos causal panel → PortfolioEnv.portfolio_risk. "
        "Docs: docs/architecture.md, docs/error_analysis.md, docs/report/outline.md, "
        "docs/morning_review.md, docs/submission_checklist.md."
    ),
    version=__version__,
    openapi_tags=OPENAPI_TAGS,
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
        endpoints=list(PUBLIC_ENDPOINTS),
        models=artifact_status(),
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
        description="synthetic source only: comma-separated synthetic benchmark names.",
    ),
    source: str = Query(
        default="auto",
        description="auto | walk_forward (committed WF results) | synthetic (smoke only)",
    ),
) -> BacktestResponse:
    names = [b.strip() for b in benchmarks.split(",") if b.strip()] or ["spy", "kospi"]
    return run_backtest(
        n_days=n_days,
        seed=seed,
        include_benchmark=include_benchmark,
        benchmarks=names,
        source=source,
    )


@app.get("/anova/results", response_model=ArtifactPayload, tags=["eval"],
         summary="ANOVA validations 1-3 computed from walk-forward results")
def get_anova_results() -> ArtifactPayload:
    return anova_results()


@app.get("/training/curves", response_model=ArtifactPayload, tags=["eval"],
         summary="PPO learning curves (Monitor) and mdd_penalty lambda sweep")
def get_training_curves() -> ArtifactPayload:
    return training_curves()


@app.get("/portfolio/history", response_model=ArtifactPayload, tags=["portfolio"],
         summary="Serving policy test-year path: cumulative, drawdown, weights, VaR/CVaR, Safe-Guard")
def get_portfolio_history() -> ArtifactPayload:
    return portfolio_history()


@app.get("/artifacts/file", tags=["eval"], summary="Serve a PNG/CSV/JSON artifact (read-only)")
def get_artifact_file(root: str = Query(..., description="drl | results"),
                      path: str = Query(..., description="relative path, e.g. shap/shap_summary.png")):
    p = safe_file(root, path)
    if p is None:
        raise HTTPException(404, f"artifact not found: {root}/{path}")
    return FileResponse(p)


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

