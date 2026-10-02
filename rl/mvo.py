"""MVO(평균-분산 최적화) 비교 기준 (요구사항 4-6).

Markowitz 평균-분산 최적화. 공분산은 252거래일 롤링 윈도우로 추정하고,
제약(비중합=1, 공매도 금지, 개별자산<=40%) 하에서 scipy로 최적화한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import MAX_ASSET_WEIGHT, SLIPPAGE, TRANSACTION_FEE

# DRL PortfolioEnv와 동일한 수수료 모델 (fee + slippage)
DEFAULT_FEE_RATE = TRANSACTION_FEE + SLIPPAGE


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


def _turnover_cost(old_w: np.ndarray, new_w: np.ndarray, fee_rate: float) -> float:
    """DRL env와 동일: one-way turnover = sum(|Δw|)/2."""
    return float(np.abs(new_w - old_w).sum()) / 2.0 * fee_rate


def rolling_mvo_backtest(
    returns_df: pd.DataFrame,
    lookback: int = 252,
    rebalance_freq: str = "ME",
    objective: str = "max_sharpe",
    fee_rate: float = DEFAULT_FEE_RATE,
) -> pd.Series:
    """월(또는 분기) 1회 리밸런싱하는 MVO 포트폴리오의 일별 수익률을 반환.

    리밸런싱 시점의 비중 추정은 `loc[:date)` (당일 수익률 제외)로 look-ahead를
    막고, 거래비용은 DRL과 같은 one-way turnover * fee_rate를 차감한다.
    """
    rebalance_dates = set(returns_df.resample(rebalance_freq).first().index)
    weights = None
    daily_returns = []

    for date in returns_df.index:
        cost = 0.0
        if date in rebalance_dates:
            # 당일 수익률을 비중 산정에 쓰지 않음 (look-ahead 방지)
            window = returns_df.loc[:date].iloc[:-1].tail(lookback)
            if len(window) >= max(30, returns_df.shape[1] * 2):
                new_weights = optimize_weights(window, objective=objective)
                if weights is None:
                    weights = np.ones(returns_df.shape[1]) / returns_df.shape[1]
                cost = _turnover_cost(weights, new_weights, fee_rate)
                weights = new_weights
        if weights is None:
            weights = np.ones(returns_df.shape[1]) / returns_df.shape[1]
        gross = float(returns_df.loc[date].to_numpy() @ weights)
        daily_returns.append(gross - cost)

    return pd.Series(daily_returns, index=returns_df.index, name="mvo_return")


def equal_weight_backtest(
    returns_df: pd.DataFrame,
    fee_rate: float = DEFAULT_FEE_RATE,
) -> pd.Series:
    """동일가중 일별 수익률. DRL env와 같이 비중 드리프트를 모델링하지 않으므로
    목표 비중이 변하지 않으면 회전율=0 → 비용 0. 수수료 모델 자체는 동일 함수를
    경유한다(초기 equal→equal도 비용 0).
    """
    n = returns_df.shape[1]
    target = np.ones(n) / n
    weights = target.copy()
    out = []
    for date in returns_df.index:
        cost = _turnover_cost(weights, target, fee_rate)
        weights = target.copy()
        gross = float(returns_df.loc[date].to_numpy() @ weights)
        out.append(gross - cost)
    return pd.Series(out, index=returns_df.index, name="equal_weight_return")
