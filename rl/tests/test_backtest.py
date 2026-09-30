import numpy as np
import pandas as pd
import pytest

from rl.backtest import compute_metrics, walk_forward_windows
from rl.mvo import optimize_weights
from rl.stats_tests import one_way_anova


@pytest.fixture
def flat_return_series():
    dates = pd.bdate_range("2020-01-01", periods=252)
    return pd.Series(0.0004, index=dates)


def test_compute_metrics_keys_count(flat_return_series):
    metrics = compute_metrics(flat_return_series)
    # 요구사항 4-5: 12개 지표
    assert len(metrics) == 12


def test_compute_metrics_positive_drift_gives_positive_cagr(flat_return_series):
    metrics = compute_metrics(flat_return_series)
    assert metrics["cagr"] > 0
    assert metrics["mdd"] >= 0


def test_mvo_weights_respect_constraints():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(300, 4)), columns=["A", "B", "C", "D"])
    weights = optimize_weights(returns, max_weight=0.4)
    assert np.isclose(weights.sum(), 1.0, atol=1e-4)
    assert (weights >= -1e-6).all()
    assert (weights <= 0.4 + 1e-6).all()


def test_walk_forward_windows_respects_data_range():
    dates = pd.bdate_range("2019-01-01", "2024-12-31")
    windows = walk_forward_windows(dates, train_years=4, test_years=1, n_windows=2)
    assert len(windows) >= 1
    for w in windows:
        assert w["train_end"] < w["test_start"]
        assert w["test_end"] <= dates.max()


def test_one_way_anova_no_difference_for_same_distribution():
    rng = np.random.default_rng(1)
    # 세 그룹 모두 동일한 분포에서 독립적으로 뽑으면 유의한 차이가 없어야 한다
    groups = {
        "a": pd.Series(rng.normal(0, 0.01, 5000)),
        "b": pd.Series(rng.normal(0, 0.01, 5000)),
        "c": pd.Series(rng.normal(0, 0.01, 5000)),
    }
    result = one_way_anova(groups)
    assert result["p_value"] > 0.05
    assert result["eta_squared"] < 0.01
