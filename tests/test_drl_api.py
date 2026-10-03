"""API wiring for the trained policy (fake model), WF results and artifacts."""
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from api.services import artifacts
from rl.data.dummy_data import generate_dummy_prices
from rl.tests.test_serving import TICKERS, fake_server

client = TestClient(app)


def _use_fake(tmp_path, monkeypatch):
    monkeypatch.setenv("DRL_ARTIFACT_DIR", str(tmp_path))
    artifacts.reset_cache()
    srv = fake_server(tmp_path)
    monkeypatch.setattr(artifacts, "_server_for", lambda root: srv)
    monkeypatch.setattr("api.services.optimize.drl_server", lambda: srv)
    monkeypatch.setattr("api.services.results.drl_server", lambda: srv)
    return srv


def test_optimize_drl_with_request_prices(tmp_path, monkeypatch):
    _use_fake(tmp_path, monkeypatch)
    prices = generate_dummy_prices(TICKERS, start="2019-01-01", end="2019-12-31").tail(120)
    body = {
        "method": "drl",
        "prices": {t: prices[t].round(4).tolist() for t in TICKERS},
        "dates": [d.strftime("%Y-%m-%d") for d in prices.index],
    }
    r = client.post("/optimize", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["method"] == "drl" and out["stub"] is False
    assert out["data_source"] == "prices" and out["as_of"] == body["dates"][-1]
    assert abs(sum(out["weights"].values()) - 1) < 1e-5
    assert len(out["per_seed"]) == 3
    assert out["latency_ms"] < 5000


def test_optimize_drl_without_model_is_503(tmp_path, monkeypatch):
    monkeypatch.setenv("DRL_ARTIFACT_DIR", str(tmp_path))
    artifacts.reset_cache()
    r = client.post("/optimize", json={"method": "drl"})
    assert r.status_code == 503
    assert "no trained DRL policy" in r.text


def test_optimize_drl_rejects_short_history(tmp_path, monkeypatch):
    _use_fake(tmp_path, monkeypatch)
    prices = generate_dummy_prices(TICKERS, start="2019-01-01", end="2019-03-01").tail(25)
    r = client.post("/optimize", json={"method": "drl", "prices": {t: prices[t].tolist() for t in TICKERS}})
    assert r.status_code == 422


def test_health_reports_model_state(tmp_path, monkeypatch):
    _use_fake(tmp_path, monkeypatch)
    m = client.get("/health").json()["models"]
    assert m["drl_model_available"] is True and m["seeds"] == [0, 1, 2]
    assert m["rag"].startswith("stub")


def test_portfolio_history_from_test_daily(tmp_path, monkeypatch):
    _use_fake(tmp_path, monkeypatch)
    r = client.get("/portfolio/history")
    d = r.json()
    assert d["available"] is True
    data = d["data"]
    assert len(data["dates"]) == 5 and data["mdd"] >= 0
    assert np.isclose(sum(data["last_weights"].values()), 1.0)


def test_backtest_default_serves_walk_forward_results():
    r = client.get("/backtest")
    assert r.status_code == 200
    d = r.json()
    assert d["source"] == "walk_forward"
    assert len(d["rows"]) == 84 and len(d["truncated_rows"]) == 19
    assert sum(1 for x in d["rows"] if x["truncated"]) == 19
    assert d["kospi_status"].startswith("SKIP")
    assert d["targets"]["meets_both"] == 0


def test_anova_results_endpoint():
    d = client.get("/anova/results").json()
    assert d["available"] is True
    assert abs(d["data"]["validation2_strategy_oneway"]["f_stat"] - 2.584) < 1e-3
    assert d["data"]["validation3_strategy_x_regime"]["status"] == "not_reproducible_here"


def test_artifact_file_rejects_traversal():
    assert client.get("/artifacts/file", params={"root": "results", "path": "../../README.md"}).status_code == 404
    ok = client.get("/artifacts/file", params={"root": "results", "path": "figures/wf_cumret_by_window.png"})
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"


def test_training_curves_unavailable_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("DRL_ARTIFACT_DIR", str(tmp_path))
    d = client.get("/training/curves").json()
    assert d["available"] is False and "Colab" in d["reason"]
