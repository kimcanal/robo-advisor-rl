"""LangGraph-shaped research pipeline: plan → retrieve → tag_risk → verify → summarize.

Mirrors Notion week-38 RAG agent (plan/exec/verify) with a plain dict state machine
so we avoid LangGraph/LangChain deps until the real stack lands.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from rag.store import InMemoryVectorStore, get_default_store
from rl.risk_tags import SCHEMA_COLUMNS, mock_risk_tags, validate_risk_tags

NodeFn = Callable[["ResearchState"], "ResearchState"]

# Allowed risk tags (must stay aligned with rl.risk_tags.mock_risk_tags)
ALLOWED_TAGS = frozenset({"earnings", "geopolitics", "credit", "liquidity", "regulatory"})


@dataclass
class Citation:
    """Placeholder citation attached to retrieved / verified evidence."""

    doc_id: str
    title: str
    source: str
    ticker: str
    score: float
    snippet: str
    quote: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "ticker": self.ticker,
            "score": self.score,
            "snippet": self.snippet,
            "quote": self.quote or self.snippet[:120],
        }


@dataclass
class ResearchState:
    """Shared state passed through graph nodes (LangGraph State analogue)."""

    query: str
    tickers: list[str]
    n_events_per_ticker: int = 3
    seed: int = 0
    top_k: int = 5
    # Plan / exec / verify artefacts
    plan: list[str] = field(default_factory=list)
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    risk_tags_df: pd.DataFrame | None = None
    verify_ok: bool = False
    verify_notes: list[str] = field(default_factory=list)
    report_excerpt: str = ""
    node_trace: list[str] = field(default_factory=list)
    node_latencies_ms: list[dict[str, Any]] = field(default_factory=list)
    store: InMemoryVectorStore | None = None


def node_plan(state: ResearchState) -> ResearchState:
    """Plan: decompose query into retrieve / tag / verify / summarize steps."""
    state.node_trace.append("plan")
    q = state.query.strip() or "market risk outlook"
    state.plan = [
        f"1. Retrieve corpus hits for query={q!r} over tickers={state.tickers}",
        "2. Execute risk tagging (rl.risk_tags schema) from retrieved context",
        "3. Verify schema + citation coverage (Self-Correction placeholder)",
        "4. Summarize report excerpt with citation placeholders for Notion",
    ]
    return state


def node_retrieve(state: ResearchState) -> ResearchState:
    """Exec retrieve: in-memory Chroma-lite query (falls back to per-ticker stubs)."""
    state.node_trace.append("retrieve")
    q = state.query.strip() or "market risk outlook"
    store = state.store or get_default_store()
    hits = store.query(q, tickers=state.tickers, top_k=state.top_k)
    # Ensure at least one stub hit per requested ticker if corpus misses it
    seen = {str(h.get("ticker", "")).upper() for h in hits}
    for t in state.tickers:
        if t.upper() not in seen:
            hits.append(
                {
                    "doc_id": f"stub-fallback-{t.lower()}",
                    "ticker": t,
                    "title": f"[stub] {q} — {t}",
                    "text": f"Synthetic snippet for {t} regarding {q!r}.",
                    "snippet": f"Synthetic snippet for {t} regarding {q!r}.",
                    "source": "stub://rag/retrieve",
                    "score": 0.01,
                    "rank": len(hits) + 1,
                    "tags": [],
                }
            )
    state.retrieved_docs = hits
    state.citations = []
    for h in hits[: state.top_k]:
        body = str(h.get("snippet") or h.get("text", "")).strip()
        # Prefer a short quote that still mentions the ticker when possible.
        quote = body[:120]
        tick = str(h.get("ticker", "")).upper()
        if tick and tick in body.upper():
            idx = body.upper().find(tick)
            quote = body[max(0, idx - 20) : max(0, idx - 20) + 120]
        tags = h.get("tags") or []
        tag_note = f" [tags={','.join(map(str, tags))}]" if tags else ""
        state.citations.append(
            Citation(
                doc_id=str(h.get("doc_id", "")),
                title=str(h.get("title", "")),
                source=str(h.get("source", "")),
                ticker=str(h.get("ticker", "")),
                score=float(h.get("score", 0.0)),
                snippet=(body[:240] + tag_note)[:280],
                quote=quote.strip() or body[:120],
            )
        )
    return state


def node_tag_risk(state: ResearchState) -> ResearchState:
    """Exec tag_risk: emit rl.risk_tags-compatible mock events (env contract)."""
    state.node_trace.append("tag_risk")
    df = mock_risk_tags(
        state.tickers,
        n_events_per_ticker=state.n_events_per_ticker,
        seed=state.seed,
    )
    # Soft bias: if a citation tag matches ALLOWED_TAGS, nudge one event tag
    cite_tags = []
    for c in state.citations:
        for doc in state.retrieved_docs:
            if doc.get("doc_id") == c.doc_id:
                cite_tags.extend(doc.get("tags") or [])
    cite_tags = [t for t in cite_tags if t in ALLOWED_TAGS]
    if cite_tags and len(df):
        df = df.copy()
        df.loc[df.index[0], "tag"] = cite_tags[0]
    state.risk_tags_df = validate_risk_tags(df)
    return state


def node_verify(state: ResearchState) -> ResearchState:
    """Verify / Self-Correction placeholder: schema + citation checks."""
    state.node_trace.append("verify")
    notes: list[str] = []
    ok = True
    if state.risk_tags_df is None or len(state.risk_tags_df) == 0:
        ok = False
        notes.append("no risk_tags produced")
    else:
        try:
            validate_risk_tags(state.risk_tags_df)
            missing = [c for c in SCHEMA_COLUMNS if c not in state.risk_tags_df.columns]
            if missing:
                ok = False
                notes.append(f"schema missing columns: {missing}")
            else:
                notes.append("risk_tags schema OK (ticker, risk_score, tag, ts)")
            bad_tags = set(state.risk_tags_df["tag"].astype(str)) - ALLOWED_TAGS
            if bad_tags:
                ok = False
                notes.append(f"unknown tags: {sorted(bad_tags)}")
            scores = state.risk_tags_df["risk_score"].astype(float)
            if ((scores < 0) | (scores > 1)).any():
                ok = False
                notes.append("risk_score out of [0, 1]")
        except Exception as exc:  # noqa: BLE001
            ok = False
            notes.append(f"schema validation failed: {exc}")

    if not state.citations:
        ok = False
        notes.append("citations empty — retrieve produced no evidence")
    else:
        notes.append(f"citations={len(state.citations)} placeholder quotes attached")

    covered = {c.ticker.upper() for c in state.citations}
    for t in state.tickers:
        if t.upper() not in covered:
            notes.append(f"warning: no citation covering ticker {t}")

    state.verify_ok = ok
    state.verify_notes = notes
    return state


def node_summarize(state: ResearchState) -> ResearchState:
    """Summarize: report excerpt + citation placeholders for Notion."""
    state.node_trace.append("summarize")
    n_tags = 0 if state.risk_tags_df is None else len(state.risk_tags_df)
    n_docs = len(state.retrieved_docs)
    tags_preview = []
    if state.risk_tags_df is not None and n_tags:
        for _, row in state.risk_tags_df.head(3).iterrows():
            tags_preview.append(f"{row['ticker']}:{row['tag']}({row['risk_score']:.2f})")
    preview = ", ".join(tags_preview) if tags_preview else "(none)"
    cite_bits = []
    for i, c in enumerate(state.citations[:3], start=1):
        cite_bits.append(f"[{i}] {c.doc_id} ({c.ticker}, score={c.score})")
    cites = "; ".join(cite_bits) if cite_bits else "(no citations)"
    verify_flag = "PASS" if state.verify_ok else "FAIL"
    # node_latencies_ms is filled by run_research_graph after each node;
    # during summarize itself the list may still be incomplete — omit if empty.
    lat_bits = []
    for item in state.node_latencies_ms:
        lat_bits.append(f"{item.get('node')}={item.get('latency_ms')}ms")
    lat_note = f" node_latencies=[{', '.join(lat_bits)}];" if lat_bits else ""
    state.report_excerpt = (
        f"[LangGraph-shaped stub plan→retrieve→tag_risk→verify→summarize] "
        f"query={state.query!r}; nodes={' → '.join(state.node_trace)};"
        f"{lat_note} "
        f"docs={n_docs}; risk_events={n_tags}; verify={verify_flag}; "
        f"sample=[{preview}]; citations={cites}. "
        "Risk tags follow rl.risk_tags env contract "
        "{ticker, risk_score[0,1], tag, ts} → PortfolioEnv portfolio_risk obs. "
        "Replace rag.graph + rag.store with real LangGraph + ChromaDB when ready."
    )
    return state


# Linear graph: plan → retrieve → tag_risk → verify → summarize
GRAPH_NODES: list[tuple[str, NodeFn]] = [
    ("plan", node_plan),
    ("retrieve", node_retrieve),
    ("tag_risk", node_tag_risk),
    ("verify", node_verify),
    ("summarize", node_summarize),
]


def _env_top_k(default: int = 5) -> int:
    raw = os.environ.get("RAG_TOP_K", "").strip()
    if not raw:
        return default
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return default


def run_research_graph(
    query: str,
    tickers: list[str],
    *,
    n_events_per_ticker: int = 3,
    seed: int = 0,
    top_k: int | None = None,
    store: InMemoryVectorStore | None = None,
) -> ResearchState:
    """Run the stub graph and return the final state.

    Env flags (optional, documented in ``rag/README.md`` / ``.env.example``):

    - ``RAG_TOP_K``: override default top_k (1–20) when ``top_k`` arg is None.
    - ``RAG_STUB_FORCE``: if truthy (``1``/``true``/``yes``), keep stub behaviour
      even when LLM keys are present (always true today — no live LLM path yet).
    - ``RAG_COLLECTION``: in-memory collection name (see ``rag.store``).
    """
    # RAG_STUB_FORCE is informational until a live LLM backend lands; always stub.
    _ = os.environ.get("RAG_STUB_FORCE", "1")
    resolved_k = _env_top_k(5) if top_k is None else top_k
    state = ResearchState(
        query=query,
        tickers=list(tickers) or ["SPY"],
        n_events_per_ticker=n_events_per_ticker,
        seed=seed,
        top_k=resolved_k,
        store=store,
    )
    for name, fn in GRAPH_NODES:
        t0 = time.perf_counter()
        state = fn(state)
        elapsed = round((time.perf_counter() - t0) * 1000.0, 3)
        state.node_latencies_ms.append({"node": name, "latency_ms": elapsed})
    return state
