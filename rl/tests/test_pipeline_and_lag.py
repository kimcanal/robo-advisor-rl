"""파이프라인 look-ahead 방지 + 피처 1일 래그 + VecNormalize freeze 스모크."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl.env.portfolio_env import PortfolioEnv
from rl.features import build_feature_frame
from rl.pipeline import prepare_env_inputs


TICKERS = ["A0", "A1", "A2"]


@pytest.fixture(scope="module")
def env_inputs():
    _, returns, rsi_df, macd_df = prepare_env_inputs(
        TICKERS, start="2019-01-01", end="2020-06-01", use_dummy=True, verbose=False
    )
    return returns, rsi_df, macd_df


def test_pipeline_does_not_zscore_macd_over_full_period(env_inputs):
    """MACD는 raw로 전달되어야 한다 (전체기간 z-score는 look-ahead).

    prepare_env_inputs가 반환한 macd_df가 build_feature_frame의 raw hist와
    인덱스 교집합 상에서 일치하는지 확인한다.
    """
    returns, rsi_df, macd_df = env_inputs
    # 더미 가격을 다시 만들어 raw MACD와 비교
    from rl.data.dummy_data import generate_dummy_prices
    from rl.features import log_returns

    prices = generate_dummy_prices(TICKERS, start="2019-01-01", end="2020-06-01")
    feature_map = build_feature_frame(prices)
    raw_macd = pd.concat({t: feature_map[t]["macd_hist"] for t in TICKERS}, axis=1)
    raw_macd = raw_macd.loc[macd_df.index]
    # 스케일링/정규화가 없어야 함 → 값 동일
    pd.testing.assert_frame_equal(
        macd_df.reset_index(drop=True),
        raw_macd.reset_index(drop=True),
        check_names=False,
        atol=1e-6,
    )
    # 전체기간 z-score였다면 평균≈0, 표준편차≈1이어야 함 — raw면 대체로 아님
    # (우연히 맞을 수 있어 약한 힌트로만; 위 equal이 본 검증)
    assert not (
        abs(macd_df.values.mean()) < 1e-3 and abs(macd_df.values.std() - 1.0) < 1e-2
    )


def test_obs_rsi_macd_use_previous_bar_not_same_as_return(env_inputs):
    """관측의 RSI/MACD는 t-1 (returns[t]와 같은 바 아님)."""
    returns, rsi_df, macd_df = env_inputs
    env = PortfolioEnv(returns, rsi_df, macd_df, window=20)
    obs, _ = env.reset()
    t = env.t  # window
    n = env.n_assets
    # obs layout: [ret_window (window*n)] + [weights (n)] + [rsi (n)] + [macd (n)]
    offset = env.window * n + n
    obs_rsi = obs[offset: offset + n]
    obs_macd = obs[offset + n: offset + 2 * n]

    expected_rsi = (rsi_df.to_numpy(dtype=np.float32) / 100.0)[t - 1]
    expected_macd = macd_df.to_numpy(dtype=np.float32)[t - 1]
    same_bar_rsi = (rsi_df.to_numpy(dtype=np.float32) / 100.0)[t]

    np.testing.assert_allclose(obs_rsi, expected_rsi, atol=1e-5)
    np.testing.assert_allclose(obs_macd, expected_macd, atol=1e-5)
    # same-bar를 썼다면 실패해야 하는 대조 (데이터가 상수인 경우만 스킵)
    if not np.allclose(expected_rsi, same_bar_rsi):
        assert not np.allclose(obs_rsi, same_bar_rsi)


def test_vecnormalize_eval_freezes_obs_rms(env_inputs):
    """평가 VecNormalize는 training=False이고 train obs_rms를 공유한다."""
    pytest.importorskip("stable_baselines3")
    from rl.vecnorm import make_eval_vecnorm, make_train_vecnorm

    returns, rsi_df, macd_df = env_inputs
    train_fn = lambda: PortfolioEnv(returns.iloc[:200], rsi_df.iloc[:200], macd_df.iloc[:200], window=20)
    train_venv = make_train_vecnorm(train_fn)
    # 통계를 조금 쌓기
    train_venv.reset()
    for _ in range(5):
        train_venv.step([train_venv.action_space.sample()])

    mean_before = train_venv.obs_rms.mean.copy()
    test_fn = lambda: PortfolioEnv(returns.iloc[150:350], rsi_df.iloc[150:350], macd_df.iloc[150:350], window=20)
    eval_venv = make_eval_vecnorm(test_fn, train_venv)
    assert eval_venv.training is False
    assert eval_venv.norm_reward is False
    eval_venv.reset()
    for _ in range(5):
        eval_venv.step([eval_venv.action_space.sample()])
    # train 통계는 eval 스텝으로 바뀌면 안 됨 (공유 객체이므로 eval이 갱신하면 같이 변함 —
    # training=False면 update_rms가 호출되지 않아야 함)
    np.testing.assert_allclose(train_venv.obs_rms.mean, mean_before)
    np.testing.assert_allclose(eval_venv.obs_rms.mean, mean_before)
