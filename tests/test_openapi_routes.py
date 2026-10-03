"""OpenAPI path contract + /health /research grader-facing fields."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api import __version__
from api.main import PUBLIC_ENDPOINTS, app

client = TestClient(app)

REQUIRED_PATHS = {
    "/health",
    "/optimize",
    "/explain",
    "/research",
    "/backtest",
    "/anova",
    "/risk-tags/apply",
    "/anova/results",
    "/training/curves",
    "/portfolio/history",
    "/artifacts/file",
}


def test_openapi_includes_required_paths():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = set(r.json().get("paths") or {})
    missing = REQUIRED_PATHS - paths
    assert not missing, f"OpenAPI missing paths: {sorted(missing)}"


def test_openapi_tags_documented():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    tags = {t["name"]: t.get("description") or "" for t in r.json().get("tags") or []}
    for name in ("ops", "portfolio", "xai", "research", "eval"):
        assert name in tags
        assert tags[name].strip(), f"empty OpenAPI tag description for {name}"


def test_health_contract_matches_version_and_endpoints():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("educational") is True
    assert isinstance(body.get("disclaimer"), str) and body["disclaimer"].strip()
    assert "not investment advice" in body["disclaimer"].lower()
    assert body.get("version") == __version__
    endpoints = body.get("endpoints") or []
    assert set(endpoints) == REQUIRED_PATHS
    assert endpoints == list(PUBLIC_ENDPOINTS)


def test_research_stub_trace_and_latencies():
    r = client.post(
        "/research",
        json={"tickers": ["SPY"], "query": "rates outlook", "n_events_per_ticker": 1},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("stub") is True
    trace = body.get("node_trace")
    assert isinstance(trace, list) and trace
    node_lats = body.get("node_latencies_ms")
    assert isinstance(node_lats, (list, dict)) and node_lats
    if isinstance(node_lats, list) and "fallback" not in trace:
        assert [x["node"] for x in node_lats] == trace
        assert all(isinstance(x.get("latency_ms"), (int, float)) for x in node_lats)
