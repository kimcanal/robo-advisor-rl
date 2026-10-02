"""Smoke tests for FastAPI health + optimize (+ research/anova stubs)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "robo-advisor-api"
    assert "version" in body


def test_optimize_equal():
    r = client.post(
        "/optimize",
        json={"tickers": ["SPY", "QQQ", "AGG"], "method": "equal"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "equal"
    assert set(body["weights"]) == {"SPY", "QQQ", "AGG"}
    assert abs(sum(body["weights"].values()) - 1.0) < 1e-5


def test_optimize_mvo_smoke():
    r = client.post(
        "/optimize",
        json={
            "tickers": ["A", "B", "C", "D"],
            "method": "mvo",
            "lookback_days": 60,
            "objective": "max_sharpe",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "mvo"
    assert len(body["weights"]) == 4
    assert abs(sum(body["weights"].values()) - 1.0) < 1e-4


def test_explain_stub():
    r = client.post("/explain", json={"tickers": ["SPY", "QQQ"], "asset_index": 0, "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["stub"] is True
    assert body["asset"] == "SPY"
    assert len(body["top_features"]) == 3


def test_research_mock_tags():
    r = client.post(
        "/research",
        json={"tickers": ["SPY"], "query": "geopolitics", "n_events_per_ticker": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["stub"] is True
    assert len(body["risk_tags"]) == 2
    for tag in body["risk_tags"]:
        assert 0.0 <= tag["risk_score"] <= 1.0
        assert tag["ticker"] == "SPY"
    # LangGraph-shaped stub wiring should show in excerpt
    assert "stub" in body["report_excerpt"].lower() or "langgraph" in body["report_excerpt"].lower()


def test_research_graph_node_trace_in_excerpt():
    r = client.post(
        "/research",
        json={"tickers": ["QQQ", "TLT"], "query": "credit stress", "n_events_per_ticker": 1},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["stub"] is True
    assert len(body["risk_tags"]) == 2
    excerpt = body["report_excerpt"].lower()
    # Prefer graph path: retrieve → tag_risk → summarize
    assert "retrieve" in excerpt or "fallback" in excerpt


def test_backtest_get():
    r = client.get("/backtest", params={"n_days": 60, "seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["n_days"] == 60
    assert "sharpe" in body["metrics"]
    assert "mdd" in body["metrics"]


def test_anova_one_way():
    r = client.post("/anova", json={"mode": "one_way", "n_obs": 80, "seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["stub"] is True
    assert body["mode"] == "one_way"
    assert "f_stat" in body["result"]
    assert "p_value" in body["result"]
    assert "eta_squared" in body["result"]
    assert set(body["groups"]) == {"drl_mdd_penalty", "mvo", "equal_weight"}


def test_anova_two_way():
    r = client.post("/anova", json={"mode": "two_way", "n_obs": 90, "seed": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["stub"] is True
    assert body["mode"] == "two_way"
    assert "anova_table" in body["result"]
    assert "eta_squared" in body["result"]
