"""Serving path: obs parity with PortfolioEnv, VecNormalize stats, seed ensemble, Safe-Guard fill."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl.backtest import compute_metrics, extend_after_safeguard
from rl.data.dummy_data import generate_dummy_prices
from rl.env.portfolio_env import PortfolioEnv
from rl.pipeline import prepare_env_inputs
from rl.serving import (
    DRLServer,
    ObsNorm,
    PolicyRun,
    action_to_weights,
    build_observation,
)

TICKERS = ["A0", "A1", "A2", "A3"]
WINDOW = 20


class _RMS:
    def __init__(self, mean, var):
        self.mean, self.var = mean, var


class FakeVecNormalize:
    """Pickle-compatible stand-in for SB3 VecNormalize (only obs_rms / clip / eps used)."""

    def __init__(self, dim, seed=0):
        rng = np.random.default_rng(seed)
        self.obs_rms = _RMS(rng.normal(0, 0.01, dim), rng.uniform(0.5, 2.0, dim))
        self.clip_obs = 10.0
        self.epsilon = 1e-8


class FakePolicy:
    """Deterministic linear 'policy' with SB3's predict signature."""

    def __init__(self, dim, n, seed):
        self.W = np.random.default_rng(seed).normal(0, 0.05, (dim, n))
        self.calls = []

    def predict(self, obs, deterministic=True):
        obs = np.atleast_2d(obs)
        self.calls.append(obs.copy())
        return np.clip(obs @ self.W, -1, 1), None


def test_build_observation_matches_env_obs():
    """API obs at date T == PortfolioEnv obs at the step whose last seen bar is T."""
    prices_full, returns, rsi, macd = prepare_env_inputs(
        TICKERS, start="2019-01-01", end="2020-12-31", use_dummy=True, verbose=False
    )
    env = PortfolioEnv(returns, rsi, macd, window=WINDOW)
    env.reset()
    raw_prices = generate_dummy_prices(TICKERS, start="2019-01-01", end="2020-12-31")
    for t in (WINDOW, WINDOW + 7, len(returns) - 1):
        env.t = t
        last_seen = returns.index[t - 1]
        obs_api, as_of = build_observation(raw_prices.loc[:last_seen], TICKERS, WINDOW)
        assert as_of == last_seen
        np.testing.assert_allclose(obs_api, env._get_obs(), rtol=1e-5, atol=1e-6)


def test_build_observation_needs_enough_history():
    raw = generate_dummy_prices(TICKERS, start="2019-01-01", end="2019-03-01")
    with pytest.raises(ValueError, match="need >="):
        build_observation(raw.iloc[:30], TICKERS, WINDOW)


def test_obsnorm_matches_vecnormalize_formula(tmp_path: Path):
    dim = 9
    vn = FakeVecNormalize(dim)
    p = tmp_path / "vecnorm.pkl"
    p.write_bytes(pickle.dumps(vn))
    norm = ObsNorm.from_vecnormalize_pickle(p)
    x = np.linspace(-1, 1, dim)
    expect = np.clip((x - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + 1e-8), -10, 10)
    np.testing.assert_allclose(norm(x), expect, rtol=1e-6)


def test_action_to_weights_matches_env_softmax():
    a = np.array([0.3, -2.0, 5.0, 0.0])
    w = action_to_weights(a)
    assert abs(w.sum() - 1) < 1e-12 and (w > 0).all()
    np.testing.assert_allclose(w, PortfolioEnv._softmax(np.clip(a, -1, 1).astype(np.float32)), rtol=1e-6)


def _write_run(root: Path, tag: str, seed: int, dim: int) -> None:
    d = root / "models" / tag
    d.mkdir(parents=True)
    (d / "model.zip").write_bytes(b"placeholder")
    (d / "vecnorm.pkl").write_bytes(pickle.dumps(FakeVecNormalize(dim, seed)))
    meta = {"run_tag": tag, "tickers": TICKERS, "window": WINDOW, "reward_type": "mdd_penalty",
            "reward_kwargs": {"mdd_lambda": 1.0}, "seed": seed, "timesteps": 120000,
            "train_start": "2015-02-01", "train_end": "2019-01-31",
            "test_start": "2019-02-01", "test_end": "2020-01-31"}
    (d / "meta.json").write_text(json.dumps(meta))
    pd.DataFrame({"date": pd.bdate_range("2019-02-01", periods=5), "daily_return": [0.01, -0.02, 0.0, 0.005, 0.01],
                  **{f"w_{t}": [0.25] * 5 for t in TICKERS}, "w_CASH": [0.0] * 5}).to_csv(d / "test_daily.csv", index=False)


def fake_server(root: Path) -> DRLServer:
    dim = WINDOW * len(TICKERS) + 3 * len(TICKERS) + 1
    for s in (0, 1, 2):
        _write_run(root, f"w1_mdd_penalty_s{s}", s, dim)
    srv = DRLServer.from_artifacts(root)
    for r in srv.runs:
        r.model = FakePolicy(dim, len(TICKERS), seed=10 + int(r.meta["seed"]))
    return srv


def test_server_seed_ensemble_is_mean_of_seed_weights(tmp_path: Path):
    srv = fake_server(tmp_path)
    assert srv.available and len(srv.runs) == 3
    prices = generate_dummy_prices(TICKERS, start="2019-01-01", end="2019-12-31")
    out = srv.predict(prices=prices)
    assert out["as_of"] == str(prices.index[-1].date())
    per = np.array([[out["per_seed"][k][t] for t in TICKERS] for k in sorted(out["per_seed"])])
    ens = np.array([out["weights"][t] for t in TICKERS])
    np.testing.assert_allclose(ens, per.mean(axis=0), rtol=1e-9)
    assert abs(ens.sum() - 1) < 1e-9
    # each seed model saw its own normalized obs
    obs, _ = build_observation(prices, TICKERS, WINDOW)
    for r in srv.runs:
        np.testing.assert_allclose(r.model.calls[-1][0], r.norm(obs), rtol=1e-6)


def test_server_without_artifacts_is_unavailable(tmp_path: Path):
    srv = DRLServer.from_artifacts(tmp_path)
    assert not srv.available
    assert srv.status()["drl_model_available"] is False
    with pytest.raises(RuntimeError):
        srv.predict()


def test_extend_after_safeguard_fills_full_year_with_cash():
    rets = np.array([0.01, -0.05, -0.12])
    full, trig = extend_after_safeguard(rets, 10, fee_rate=0.00065, risk_free=0.04)
    assert len(full) == 10 and trig == 2
    daily_rf = np.log1p(0.04) / 252
    assert full[3] == pytest.approx(daily_rf - 0.00065)
    np.testing.assert_allclose(full[4:], daily_rf)
    m = compute_metrics(pd.Series(full))
    from rl.analysis.wf_analysis import implied_test_days

    assert implied_test_days(m["cumulative_return"], m["cagr"]) == pytest.approx(10, abs=1e-6)
    same, none = extend_after_safeguard(np.zeros(10), 10, 0.001)
    assert none is None and len(same) == 10


def test_policy_run_weights_uses_norm_then_softmax(tmp_path: Path):
    dim = WINDOW * len(TICKERS) + 3 * len(TICKERS) + 1
    norm = ObsNorm(mean=np.zeros(dim), var=np.ones(dim))
    run = PolicyRun(run_dir=tmp_path, meta={"tickers": TICKERS, "window": WINDOW}, norm=norm,
                    model=FakePolicy(dim, len(TICKERS), 3))
    x = np.random.default_rng(0).normal(size=dim).astype(np.float32)
    w = run.weights(x)
    a, _ = run.model.predict(norm(x)[None, :])
    np.testing.assert_allclose(w, action_to_weights(a[0]))
