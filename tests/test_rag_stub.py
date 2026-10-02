"""Unit tests for LangGraph-shaped rag stub (no LLM keys)."""
from __future__ import annotations

from rag.graph import GRAPH_NODES, run_research_graph
from rl.risk_tags import SCHEMA_COLUMNS, validate_risk_tags


def test_graph_nodes_order():
    assert [name for name, _ in GRAPH_NODES] == ["retrieve", "tag_risk", "summarize"]


def test_run_research_graph_returns_valid_tags():
    state = run_research_graph(
        query="liquidity risk",
        tickers=["SPY", "AGG"],
        n_events_per_ticker=2,
        seed=42,
    )
    assert state.node_trace == ["retrieve", "tag_risk", "summarize"]
    assert len(state.retrieved_docs) == 2
    assert state.risk_tags_df is not None
    validated = validate_risk_tags(state.risk_tags_df)
    assert list(validated.columns) == list(SCHEMA_COLUMNS)
    assert len(validated) == 4
    assert "retrieve" in state.report_excerpt
    assert "summarize" in state.report_excerpt or "→" in state.report_excerpt
