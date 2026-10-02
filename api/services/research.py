"""Research / RAG stub — returns mock risk tags matching rl.risk_tags schema."""
from __future__ import annotations

from api.schemas import ResearchRequest, ResearchResponse, RiskTag
from rl.risk_tags import mock_risk_tags


def run_research(req: ResearchRequest) -> ResearchResponse:
    tickers = list(req.tickers) or ["SPY"]
    df = mock_risk_tags(
        tickers,
        n_events_per_ticker=req.n_events_per_ticker,
        seed=abs(hash(req.query)) % (2**32),
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
    excerpt = (
        f"[stub RAG] Query={req.query!r}. Generated {len(tags)} mock risk events "
        f"for {tickers}. Replace with LangGraph/Chroma pipeline when ready."
    )
    return ResearchResponse(query=req.query, risk_tags=tags, report_excerpt=excerpt, stub=True)


def pd_ts(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)
