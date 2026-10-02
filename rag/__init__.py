"""LangGraph-shaped RAG research agent stub (plain Python state machine).

Real LangGraph can replace `rag.graph.run_research_graph` later without changing
the API contract (`risk_tags` schema aligned with `rl.risk_tags`).
No LLM API keys required for this educational stub.
"""

from rag.graph import ResearchState, run_research_graph

__all__ = ["ResearchState", "run_research_graph"]
