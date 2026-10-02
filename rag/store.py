"""In-memory Chroma-lite vector store (no chromadb / LLM dependency).

Educational stand-in for Notion week-38 RAG stack: bag-of-words + cosine
similarity over a seeded corpus of market-risk snippets. Swap for ChromaDB
when the research team lands real embeddings.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+", re.I)

# Seed corpus: synthetic "news" snippets keyed for demo retrieval/citation.
_SEED_DOCS: list[dict[str, Any]] = [
    {
        "doc_id": "stub-spy-liquidity-01",
        "ticker": "SPY",
        "title": "US equity liquidity watch",
        "text": (
            "S&P 500 ETF SPY saw thinner order books and higher bid-ask spreads amid "
            "liquidity stress and volatility spikes in risk-off sessions."
        ),
        "source": "stub://corpus/liquidity",
        "tags": ["liquidity", "volatility"],
    },
    {
        "doc_id": "stub-spy-geopolitics-01",
        "ticker": "SPY",
        "title": "Geopolitical risk and equity beta",
        "text": (
            "Geopolitics headlines lifted equity risk premia; SPY drawdowns clustered "
            "around escalation events historically used in educational demos."
        ),
        "source": "stub://corpus/geopolitics",
        "tags": ["geopolitics", "equity"],
    },
    {
        "doc_id": "stub-qqq-earnings-01",
        "ticker": "QQQ",
        "title": "Tech earnings concentration",
        "text": (
            "Nasdaq-100 QQQ sensitivity to mega-cap earnings surprises and regulatory "
            "scrutiny remains elevated versus broad market baskets."
        ),
        "source": "stub://corpus/earnings",
        "tags": ["earnings", "regulatory"],
    },
    {
        "doc_id": "stub-tlt-credit-01",
        "ticker": "TLT",
        "title": "Duration and credit spillover",
        "text": (
            "Long Treasuries TLT reacted to credit-spread widening and flight-to-quality "
            "flows; educational stub for rate and credit risk tagging."
        ),
        "source": "stub://corpus/credit",
        "tags": ["credit", "rates"],
    },
    {
        "doc_id": "stub-agg-rates-01",
        "ticker": "AGG",
        "title": "Investment-grade bond outlook",
        "text": (
            "AGG aggregate bond ETF tracks duration and investment-grade credit; stub "
            "snippet for regulatory and rates risk narratives."
        ),
        "source": "stub://corpus/rates",
        "tags": ["rates", "regulatory"],
    },
    {
        "doc_id": "stub-eem-geopolitics-01",
        "ticker": "EEM",
        "title": "EM FX and geopolitics",
        "text": (
            "Emerging markets EEM face FX liquidity shocks tied to geopolitics and "
            "capital-flow reversals in the educational corpus."
        ),
        "source": "stub://corpus/em",
        "tags": ["geopolitics", "liquidity"],
    },
    {
        "doc_id": "stub-hyg-credit-01",
        "ticker": "HYG",
        "title": "High-yield credit stress",
        "text": (
            "HYG high-yield spreads widened under credit stress scenarios used for "
            "mock risk-tag generation in the RAG stub."
        ),
        "source": "stub://corpus/hyg",
        "tags": ["credit", "liquidity"],
    },
    {
        "doc_id": "stub-gld-macro-01",
        "ticker": "GLD",
        "title": "Gold as macro hedge stub",
        "text": (
            "GLD often cited as a geopolitics and inflation hedge; stub document for "
            "citation placeholders in research reports."
        ),
        "source": "stub://corpus/gld",
        "tags": ["geopolitics", "macro"],
    },
    {
        "doc_id": "stub-bil-liquidity-01",
        "ticker": "BIL",
        "title": "T-bill ETF liquidity stub",
        "text": (
            "BIL short Treasury ETF used as a cash / risk-free proxy; educational stub "
            "for liquidity and rates narratives when risk-off flows hit equities."
        ),
        "source": "stub://corpus/bil",
        "tags": ["liquidity", "rates"],
    },
    {
        "doc_id": "stub-vnq-regulatory-01",
        "ticker": "VNQ",
        "title": "REIT regulatory and rate sensitivity",
        "text": (
            "VNQ real-estate ETF reacts to regulatory shifts and duration; stub snippet "
            "for regulatory risk tagging in the RAG demo corpus."
        ),
        "source": "stub://corpus/vnq",
        "tags": ["regulatory", "rates"],
    },
    {
        "doc_id": "stub-spy-earnings-01",
        "ticker": "SPY",
        "title": "Broad-market earnings season stub",
        "text": (
            "SPY tracks aggregated mega-cap earnings surprises; stub document so "
            "citation coverage includes earnings tags for the broad market ETF."
        ),
        "source": "stub://corpus/earnings-spy",
        "tags": ["earnings", "volatility"],
    },
    {
        "doc_id": "stub-qqq-regulatory-01",
        "ticker": "QQQ",
        "title": "Tech regulatory scrutiny stub",
        "text": (
            "QQQ faces regulatory headlines around competition and data privacy; "
            "educational corpus entry for regulatory risk_score demos."
        ),
        "source": "stub://corpus/regulatory-qqq",
        "tags": ["regulatory", "earnings"],
    },
]


def _tokenize(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for tok in _TOKEN.findall(text.lower()):
        counts[tok] = counts.get(tok, 0.0) + 1.0
    return counts


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(av * b[k] for k, av in a.items() if k in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def _stable_id(payload: dict[str, Any]) -> str:
    blob = repr(sorted(payload.items())).encode("utf-8")
    return "stub-" + hashlib.sha1(blob).hexdigest()[:10]


@dataclass
class InMemoryVectorStore:
    """Minimal Chroma-lite: add documents, query by cosine over bag-of-words."""

    collection_name: str = "robo_advisor_stub"
    _docs: list[dict[str, Any]] = field(default_factory=list)
    _vectors: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._docs:
            self.seed_default_corpus()

    def clear(self) -> None:
        self._docs.clear()
        self._vectors.clear()

    def seed_default_corpus(self) -> int:
        self.clear()
        for doc in _SEED_DOCS:
            self.add(doc)
        return len(self._docs)

    def add(self, doc: dict[str, Any]) -> str:
        payload = dict(doc)
        doc_id = str(payload.get("doc_id") or _stable_id(payload))
        payload["doc_id"] = doc_id
        text = " ".join(
            str(payload.get(k, ""))
            for k in ("title", "text", "ticker", "tags", "source")
        )
        self._docs.append(payload)
        self._vectors.append(_tokenize(text))
        return doc_id

    def count(self) -> int:
        return len(self._docs)

    def query(
        self,
        query: str,
        *,
        tickers: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        qvec = _tokenize(query)
        ticker_set = {t.upper() for t in (tickers or [])}
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc, vec in zip(self._docs, self._vectors):
            score = _cosine(qvec, vec)
            tick = str(doc.get("ticker", "")).upper()
            if ticker_set and tick in ticker_set:
                score += 0.15
            tags = doc.get("tags") or []
            for tag in tags:
                if str(tag).lower() in query.lower():
                    score += 0.05
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for rank, (score, doc) in enumerate(scored[: max(1, top_k)], start=1):
            hit = dict(doc)
            hit["score"] = round(float(score), 4)
            hit["rank"] = rank
            hit["snippet"] = str(doc.get("text", ""))[:240]
            out.append(hit)
        return out


# Process-wide default store (lazy-friendly singleton for API/graph).
_DEFAULT_STORE: InMemoryVectorStore | None = None


def get_default_store() -> InMemoryVectorStore:
    """Return process-wide store; collection_name from ``RAG_COLLECTION`` if set."""
    import os

    global _DEFAULT_STORE
    name = (os.environ.get("RAG_COLLECTION") or "robo_advisor_stub").strip() or "robo_advisor_stub"
    if _DEFAULT_STORE is None or _DEFAULT_STORE.collection_name != name:
        _DEFAULT_STORE = InMemoryVectorStore(collection_name=name)
    return _DEFAULT_STORE


def reset_default_store() -> InMemoryVectorStore:
    """Drop singleton (tests / env flag changes)."""
    global _DEFAULT_STORE
    _DEFAULT_STORE = None
    return get_default_store()
