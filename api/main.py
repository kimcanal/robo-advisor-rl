"""FastAPI entrypoint — /health /optimize /explain /research /backtest."""
from __future__ import annotations

from fastapi import FastAPI, Query

from api import __version__
from api.schemas import (
    BacktestResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    OptimizeRequest,
    OptimizeResponse,
    ResearchRequest,
    ResearchResponse,
)
from api.services.backtest_svc import run_backtest
from api.services.explain import run_explain
from api.services.optimize import run_optimize
from api.services.research import run_research

app = FastAPI(
    title="Robo-Advisor API",
    description=(
        "Educational skeleton wrapping RL / research modules. "
        "Not investment advice; backtests ≠ future returns."
    ),
    version=__version__,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(version=__version__)


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
) -> BacktestResponse:
    return run_backtest(n_days=n_days, seed=seed, include_benchmark=include_benchmark)
