"""Research / RAG service — wires LangGraph-shaped stub (plan/exec/verify)."""
from __future__ import annotations

import time

from api.schemas import Citation, ResearchRequest, ResearchResponse, RiskTag


def pd_ts(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def run_research(req: ResearchRequest) -> ResearchResponse:
    t0 = time.perf_counter()
    tickers = list(req.tickers) or ["SPY"]
    seed = abs(hash(req.query)) % (2**32)

    plan: list[str] = []
    node_trace: list[str] = []
    citations: list[Citation] = []
    verify_ok = False
    verify_notes: list[str] = []

    try:
        from rag.graph import run_research_graph

        state = run_research_graph(
            query=req.query,
            tickers=tickers,
            n_events_per_ticker=req.n_events_per_ticker,
            seed=seed,
            top_k=req.top_k,
        )
        df = state.risk_tags_df
        excerpt = state.report_excerpt
        plan = list(state.plan)
        node_trace = list(state.node_trace)
        verify_ok = bool(state.verify_ok)
        verify_notes = list(state.verify_notes)
        citations = [
            Citation(
                doc_id=c.doc_id,
                title=c.title,
                source=c.source,
                ticker=c.ticker,
                score=float(c.score),
                snippet=c.snippet,
                quote=c.quote,
            )
            for c in state.citations
        ]
    except Exception as exc:  # noqa: BLE001 — keep API up if stub import fails
        from rl.risk_tags import mock_risk_tags

        df = mock_risk_tags(
            tickers,
            n_events_per_ticker=req.n_events_per_ticker,
            seed=seed,
        )
        excerpt = (
            f"[fallback stub] Query={req.query!r}. Generated mock risk events "
            f"for {tickers} (rag graph unavailable: {exc})."
        )
        plan = ["fallback: mock_risk_tags only"]
        node_trace = ["fallback"]
        verify_notes = [f"rag graph unavailable: {exc}"]

    tags = [
        RiskTag(
            ticker=str(row["ticker"]),
            risk_score=float(row["risk_score"]),
            tag=str(row["tag"]),
            ts=str(pd_ts(row["ts"])),
        )
        for _, row in df.iterrows()
    ]
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return ResearchResponse(
        query=req.query,
        risk_tags=tags,
        report_excerpt=excerpt,
        stub=True,
        plan=plan,
        node_trace=node_trace,
        citations=citations,
        verify_ok=verify_ok,
        verify_notes=verify_notes,
        latency_ms=latency_ms,
    )
