import numpy as np
import pandas as pd
import pytest

from rl.backtest import compute_metrics, walk_forward_windows
from rl.mvo import equal_weight_backtest, optimize_weights, rolling_mvo_backtest
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


def test_mvo_excludes_same_day_return_when_sizing():
    """리밸런스일 비중 산정에 당일 수익률이 들어가면 안 된다.

    조작: 리밸런스일에만 특정 자산이 폭등하면, look-ahead가 있으면 그 자산을
    과대비중 → 당일 수익이 비정상적으로 큼. 수정 후에는 그렇지 않음.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=300)
    rets = pd.DataFrame(rng.normal(0, 0.001, size=(300, 3)), index=dates, columns=["A", "B", "C"])
    # 매월 첫 거래일에 A만 +50% (비정상) — look-ahead면 그날을 A에 몰빵
    month_starts = rets.resample("ME").first().index
    for d in month_starts:
        if d in rets.index:
            rets.loc[d, "A"] = 0.50
            rets.loc[d, ["B", "C"]] = -0.01
    series = rolling_mvo_backtest(rets, lookback=60, fee_rate=0.0)
    # 당일 수익을 sizing에 쓰지 않으므로, 리밸런스일의 포트 수익이 0.5에 가깝지 않아야 함
    # (동일가중이면 ~0.16). look-ahead면 ~0.5에 가까움.
    rebalance_rets = series.loc[series.index.isin(month_starts)]
    assert rebalance_rets.max() < 0.35


def test_equal_weight_uses_same_fee_model_zero_when_static():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(50, 4)), columns=list("ABCD"))
    with_fee = equal_weight_backtest(rets, fee_rate=0.001)
    no_fee = equal_weight_backtest(rets, fee_rate=0.0)
    # 비중 드리프트 미모델 → 목표 비중 불변 → 비용 0
    pd.testing.assert_series_equal(with_fee, no_fee)
