"""Minimal graph-shaped research pipeline: retrieve → tag_risk → summarize.

Mirrors LangGraph node/edge concepts with a plain dict state machine so we avoid
pulling in heavy LangGraph/LangChain deps until the real RAG stack lands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from rl.risk_tags import mock_risk_tags, validate_risk_tags

NodeFn = Callable[["ResearchState"], "ResearchState"]


@dataclass
class ResearchState:
    """Shared state passed through graph nodes (LangGraph State analogue)."""

    query: str
    tickers: list[str]
    n_events_per_ticker: int = 3
    seed: int = 0
    # Node outputs
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    risk_tags_df: pd.DataFrame | None = None
    report_excerpt: str = ""
    node_trace: list[str] = field(default_factory=list)


def node_retrieve(state: ResearchState) -> ResearchState:
    """Stub retrieve: fake document hits keyed by query + tickers."""
    state.node_trace.append("retrieve")
    q = state.query.strip() or "market risk outlook"
    state.retrieved_docs = [
        {
            "ticker": t,
            "title": f"[stub] {q} — {t}",
            "snippet": f"Synthetic snippet for {t} regarding {q!r}.",
            "source": "stub://rag/retrieve",
        }
        for t in state.tickers
    ]
    return state


def node_tag_risk(state: ResearchState) -> ResearchState:
    """Stub tag_risk: emit rl.risk_tags-compatible mock events."""
    state.node_trace.append("tag_risk")
    df = mock_risk_tags(
        state.tickers,
        n_events_per_ticker=state.n_events_per_ticker,
        seed=state.seed,
    )
    state.risk_tags_df = validate_risk_tags(df)
    return state


def node_summarize(state: ResearchState) -> ResearchState:
    """Stub summarize: short report excerpt from tags + retrieved docs."""
    state.node_trace.append("summarize")
    n_tags = 0 if state.risk_tags_df is None else len(state.risk_tags_df)
    n_docs = len(state.retrieved_docs)
    tags_preview = []
    if state.risk_tags_df is not None and n_tags:
        for _, row in state.risk_tags_df.head(3).iterrows():
            tags_preview.append(f"{row['ticker']}:{row['tag']}({row['risk_score']:.2f})")
    preview = ", ".join(tags_preview) if tags_preview else "(none)"
    state.report_excerpt = (
        f"[LangGraph-shaped stub] query={state.query!r}; "
        f"nodes={' → '.join(state.node_trace)}; "
        f"docs={n_docs}; risk_events={n_tags}; sample=[{preview}]. "
        "Replace rag.graph with real LangGraph + vector store when ready."
    )
    return state


# Linear graph: retrieve → tag_risk → summarize (LangGraph edge analogue).
GRAPH_NODES: list[tuple[str, NodeFn]] = [
    ("retrieve", node_retrieve),
    ("tag_risk", node_tag_risk),
    ("summarize", node_summarize),
]


def run_research_graph(
    query: str,
    tickers: list[str],
    *,
    n_events_per_ticker: int = 3,
    seed: int = 0,
) -> ResearchState:
    """Run the stub graph and return the final state."""
    state = ResearchState(
        query=query,
        tickers=list(tickers) or ["SPY"],
        n_events_per_ticker=n_events_per_ticker,
        seed=seed,
    )
    for _name, fn in GRAPH_NODES:
        state = fn(state)
    return state
