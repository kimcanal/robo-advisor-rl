"""Unit tests for LangGraph-shaped rag stub (plan/exec/verify, no LLM keys)."""
from __future__ import annotations

from rag.graph import GRAPH_NODES, run_research_graph
from rag.store import InMemoryVectorStore, get_default_store
from rl.risk_tags import SCHEMA_COLUMNS, validate_risk_tags


def test_graph_nodes_order():
    assert [name for name, _ in GRAPH_NODES] == [
        "plan",
        "retrieve",
        "tag_risk",
        "verify",
        "summarize",
    ]


def test_run_research_graph_returns_valid_tags():
    state = run_research_graph(
        query="liquidity risk",
        tickers=["SPY", "AGG"],
        n_events_per_ticker=2,
        seed=42,
    )
    assert state.node_trace == ["plan", "retrieve", "tag_risk", "verify", "summarize"]
    assert len(state.plan) >= 3
    assert len(state.retrieved_docs) >= 2
    assert state.risk_tags_df is not None
    validated = validate_risk_tags(state.risk_tags_df)
    assert list(validated.columns) == list(SCHEMA_COLUMNS)
    assert len(validated) == 4
    assert state.verify_ok is True
    assert len(state.citations) >= 1
    assert "retrieve" in state.report_excerpt
    assert "verify" in state.report_excerpt.lower() or "PASS" in state.report_excerpt


def test_inmemory_store_query_prefers_ticker():
    store = InMemoryVectorStore()
    assert store.count() >= 5
    hits = store.query("liquidity stress", tickers=["SPY"], top_k=3)
    assert hits
    assert hits[0]["score"] >= hits[-1]["score"]
    # default singleton also works
    assert get_default_store().count() >= 5


def test_citations_have_placeholder_quote():
    state = run_research_graph(
        query="geopolitics",
        tickers=["SPY"],
        n_events_per_ticker=1,
        seed=1,
        top_k=3,
    )
    assert state.citations
    c0 = state.citations[0]
    assert c0.doc_id
    assert c0.source
    assert c0.quote or c0.snippet


def test_seed_corpus_covers_extra_tickers():
    store = InMemoryVectorStore()
    tickers = {str(d.get("ticker", "")).upper() for d in store._docs}
    for t in ("BIL", "VNQ", "SPY", "QQQ"):
        assert t in tickers
    assert store.count() >= 10


def test_rag_top_k_env(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "2")
    state = run_research_graph(
        query="liquidity",
        tickers=["SPY"],
        n_events_per_ticker=1,
        seed=0,
        top_k=None,
    )
    assert state.top_k == 2
    assert len(state.citations) <= 2


def test_rag_collection_env(monkeypatch):
    from rag.store import reset_default_store

    monkeypatch.setenv("RAG_COLLECTION", "week38_demo")
    store = reset_default_store()
    assert store.collection_name == "week38_demo"
    assert store.count() >= 8


def test_citation_snippet_may_include_tags():
    state = run_research_graph(
        query="liquidity stress",
        tickers=["SPY"],
        n_events_per_ticker=1,
        seed=2,
        top_k=3,
    )
    assert state.verify_ok is True
    # At least one citation should carry a non-empty quote
    assert any((c.quote or c.snippet) for c in state.citations)
