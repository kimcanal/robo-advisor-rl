"""리스크 태그 로드 + PortfolioEnv 관측 축 스모크 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl.env.portfolio_env import PortfolioEnv
from rl.pipeline import prepare_env_inputs
from rl.risk_tags import (
    RiskTagSchemaError,
    load_risk_tags_csv,
    mock_risk_tags,
    risk_score_panel,
    save_risk_tags_csv,
    validate_risk_tags,
)

TICKERS = ["A0", "A1", "A2"]


@pytest.fixture(scope="module")
def env_inputs():
    _, returns, rsi_df, macd_df = prepare_env_inputs(
        TICKERS, start="2019-01-01", end="2020-06-01", use_dummy=True, verbose=False
    )
    return returns, rsi_df, macd_df


def test_mock_and_csv_roundtrip(tmp_path):
    tags = mock_risk_tags(TICKERS, start="2019-01-01", end="2019-06-30", seed=7)
    assert list(tags.columns) == ["ticker", "risk_score", "tag", "ts"]
    assert tags["risk_score"].between(0, 1).all()

    path = tmp_path / "risk_tags.csv"
    save_risk_tags_csv(tags, path)
    loaded = load_risk_tags_csv(path)
    pd.testing.assert_frame_equal(
        tags.reset_index(drop=True),
        loaded.reset_index(drop=True),
        check_dtype=False,
    )


def test_schema_rejects_bad_score():
    bad = pd.DataFrame(
        [{"ticker": "A0", "risk_score": 1.5, "tag": "x", "ts": "2020-01-02"}]
    )
    with pytest.raises(RiskTagSchemaError):
        validate_risk_tags(bad)


def test_risk_score_panel_is_causal():
    """이벤트 이후 날짜에만 score가 반영되고, 이전은 0."""
    tags = pd.DataFrame(
        [
            {"ticker": "A0", "risk_score": 0.8, "tag": "earn", "ts": "2020-02-05"},
            {"ticker": "A1", "risk_score": 0.4, "tag": "geo", "ts": "2020-02-07"},
        ]
    )
    dates = pd.bdate_range("2020-02-03", "2020-02-14")  # Mon..Fri business days
    panel = risk_score_panel(tags, ["A0", "A1"], dates)
    assert panel.loc["2020-02-04", "A0"] == pytest.approx(0.0)
    assert panel.loc["2020-02-05", "A0"] == pytest.approx(0.8)
    assert panel.loc["2020-02-06", "A1"] == pytest.approx(0.0)
    assert panel.loc["2020-02-07", "A1"] == pytest.approx(0.4)


def test_obs_shape_backward_compatible_without_tags(env_inputs):
    returns, rsi_df, macd_df = env_inputs
    env = PortfolioEnv(returns, rsi_df, macd_df, window=20)
    obs, _ = env.reset()
    expected = 20 * 3 + 3 * 3 + PortfolioEnv.N_RISK_OBS
    assert env.observation_space.shape == (expected,)
    assert obs.shape == (expected,)
    # 태그 없으면 리스크 축은 0
    assert obs[-1] == pytest.approx(0.0)


def test_obs_includes_nonzero_risk_with_tags(env_inputs):
    returns, rsi_df, macd_df = env_inputs
    tags = mock_risk_tags(TICKERS, start=str(returns.index.min().date()), end=str(returns.index.max().date()), seed=1)
    env = PortfolioEnv(returns, rsi_df, macd_df, window=20, risk_tags=tags)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    # 이벤트 이후 구간에서 step 하면 portfolio_risk / info에 반영
    saw_risk = False
    for _ in range(40):
        obs, _, terminated, truncated, info = env.step(env.action_space.sample())
        if info["portfolio_risk"] > 0:
            saw_risk = True
            assert "effective_mdd_limit" in info
            break
        if terminated or truncated:
            break
    assert saw_risk


def test_risk_mdd_scale_tightens_limit(env_inputs):
    returns, rsi_df, macd_df = env_inputs
    # 전 구간 고위험 패널
    panel = pd.DataFrame(0.9, index=returns.index, columns=list(returns.columns))
    env = PortfolioEnv(
        returns, rsi_df, macd_df, window=20, risk_panel=panel, risk_mdd_scale=0.5, mdd_limit=0.15
    )
    env.reset()
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    # effective = 0.15 * (1 - 0.5*0.9) = 0.15 * 0.55 = 0.0825
    assert info["effective_mdd_limit"] == pytest.approx(0.15 * (1 - 0.5 * 0.9), rel=1e-5)
    assert info["effective_mdd_limit"] < 0.15
