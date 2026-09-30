import numpy as np
import pytest

from rl.rewards import get_reward_fn, reward_mdd_penalty, reward_sharpe, reward_simple


def test_reward_simple_passthrough():
    assert reward_simple(0.05, []) == 0.05
    assert reward_simple(-0.02, [0.01, 0.02]) == -0.02


def test_reward_sharpe_scales_by_volatility():
    low_vol_history = [0.001] * 20
    high_vol_history = [0.05, -0.05] * 10
    r_low = reward_sharpe(0.01, low_vol_history)
    r_high = reward_sharpe(0.01, high_vol_history)
    # 동일 수익률이라도 변동성이 크면 보상이 깎여야 한다
    assert r_low > r_high


def test_reward_mdd_penalty_punishes_drawdown():
    # 계속 손실이 누적된 상태에서의 보상은 순수익률보다 작아야 한다 (페널티 적용)
    history = [-0.02, -0.02, -0.02]
    r = reward_mdd_penalty(-0.01, history, mdd_lambda=1.0)
    assert r < -0.01


def test_get_reward_fn_invalid_name_raises():
    with pytest.raises(ValueError):
        get_reward_fn("does_not_exist")
