import numpy as np
import pytest

from rl.env.portfolio_env import PortfolioEnv
from rl.pipeline import prepare_env_inputs

TICKERS = ["A0", "A1", "A2"]


@pytest.fixture(scope="module")
def env_inputs():
    _, returns, rsi_df, macd_df = prepare_env_inputs(
        TICKERS, start="2019-01-01", end="2020-06-01", use_dummy=True, verbose=False
    )
    return returns, rsi_df, macd_df


def test_reset_returns_correct_obs_shape(env_inputs):
    returns, rsi_df, macd_df = env_inputs
    env = PortfolioEnv(returns, rsi_df, macd_df, window=20)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)


def test_step_weights_sum_to_one(env_inputs):
    returns, rsi_df, macd_df = env_inputs
    env = PortfolioEnv(returns, rsi_df, macd_df, window=20)
    env.reset()
    action = env.action_space.sample()
    env.step(action)
    assert np.isclose(env.weights.sum(), 1.0, atol=1e-5)
    assert (env.weights >= 0).all()


def test_no_shorting_even_with_extreme_action(env_inputs):
    returns, rsi_df, macd_df = env_inputs
    env = PortfolioEnv(returns, rsi_df, macd_df, window=20)
    env.reset()
    extreme_action = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    env.step(extreme_action)
    assert (env.weights >= 0).all()
    assert np.isclose(env.weights.sum(), 1.0, atol=1e-5)


def test_safe_guard_triggers_on_large_drawdown(env_inputs):
    returns, rsi_df, macd_df = env_inputs
    env = PortfolioEnv(returns, rsi_df, macd_df, window=20, mdd_limit=0.0001)
    env.reset()
    terminated = False
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    assert terminated
    assert info["drawdown"] > env.mdd_limit


def test_episode_ends_at_data_boundary(env_inputs):
    returns, rsi_df, macd_df = env_inputs
    env = PortfolioEnv(returns, rsi_df, macd_df, window=20, mdd_limit=1.0)  # Safe-Guard 사실상 비활성화
    env.reset()
    steps = 0
    done = False
    static_action = np.zeros(3, dtype=np.float32)
    while not done and steps < 10_000:
        _, _, terminated, truncated, _ = env.step(static_action)
        done = terminated or truncated
        steps += 1
    assert done
    assert steps == env.n_steps - env.window
