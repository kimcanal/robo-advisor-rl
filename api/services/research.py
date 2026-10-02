"""Research / RAG service — wires LangGraph-shaped stub when available."""
from __future__ import annotations

from api.schemas import ResearchRequest, ResearchResponse, RiskTag


def pd_ts(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def run_research(req: ResearchRequest) -> ResearchResponse:
    tickers = list(req.tickers) or ["SPY"]
    seed = abs(hash(req.query)) % (2**32)

    # Prefer rag.graph stub (retrieve → tag_risk → summarize); fall back to mock tags.
    try:
        from rag.graph import run_research_graph

        state = run_research_graph(
            query=req.query,
            tickers=tickers,
            n_events_per_ticker=req.n_events_per_ticker,
            seed=seed,
        )
        df = state.risk_tags_df
        excerpt = state.report_excerpt
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

    tags = [
        RiskTag(
            ticker=str(row["ticker"]),
            risk_score=float(row["risk_score"]),
            tag=str(row["tag"]),
            ts=str(pd_ts(row["ts"])),
        )
        for _, row in df.iterrows()
    ]
    return ResearchResponse(
        query=req.query,
        risk_tags=tags,
        report_excerpt=excerpt,
        stub=True,
    )
