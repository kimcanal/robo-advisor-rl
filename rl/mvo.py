"""MVO(평균-분산 최적화) 비교 기준 (요구사항 4-6).

Markowitz 평균-분산 최적화. 공분산은 252거래일 롤링 윈도우로 추정하고,
제약(비중합=1, 공매도 금지, 개별자산<=40%) 하에서 scipy로 최적화한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import MAX_ASSET_WEIGHT


def _neg_sharpe(weights: np.ndarray, mean_ret: np.ndarray, cov: np.ndarray) -> float:
    port_ret = weights @ mean_ret
    port_vol = np.sqrt(weights @ cov @ weights)
    if port_vol < 1e-8:
        return 0.0
    return -(port_ret / port_vol)


def _min_variance(weights: np.ndarray, mean_ret: np.ndarray, cov: np.ndarray) -> float:
    return weights @ cov @ weights


def optimize_weights(
    returns_window: pd.DataFrame,
    objective: str = "max_sharpe",
    max_weight: float = MAX_ASSET_WEIGHT,
) -> np.ndarray:
    """주어진 수익률 윈도우로 MVO 비중을 계산한다.

    objective: "max_sharpe" 또는 "min_variance" 중 선택.
    선택 근거: max_sharpe는 위험조정수익 극대화가 목적일 때, min_variance는
    변동성 자체를 최소화하고 싶을 때 적합 -> RL(수익-리스크 결합 보상)과
    비교축을 맞추기 위해 기본값은 max_sharpe로 둔다.
    """
    n = returns_window.shape[1]
    mean_ret = returns_window.mean().to_numpy()
    cov = returns_window.cov().to_numpy()

    obj_fn = _neg_sharpe if objective == "max_sharpe" else _min_variance
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, max_weight) for _ in range(n)]
    x0 = np.ones(n) / n

    result = minimize(
        obj_fn, x0, args=(mean_ret, cov), method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not result.success:
        return x0  # 수렴 실패 시 동일가중으로 폴백 (리포트에 명시할 것)
    return result.x


def rolling_mvo_backtest(
    returns_df: pd.DataFrame,
    lookback: int = 252,
    rebalance_freq: str = "ME",
    objective: str = "max_sharpe",
) -> pd.Series:
    """월(또는 분기) 1회 리밸런싱하는 MVO 포트폴리오의 일별 수익률을 반환."""
    rebalance_dates = returns_df.resample(rebalance_freq).first().index
    weights = None
    daily_returns = []

    for date in returns_df.index:
        if date in rebalance_dates:
            window = returns_df.loc[:date].tail(lookback)
            if len(window) >= max(30, returns_df.shape[1] * 2):
                weights = optimize_weights(window, objective=objective)
        if weights is None:
            weights = np.ones(returns_df.shape[1]) / returns_df.shape[1]
        daily_returns.append(float(returns_df.loc[date].to_numpy() @ weights))

    return pd.Series(daily_returns, index=returns_df.index, name="mvo_return")
