"""성과 지표 12종 + Walk-Forward 분할 (요구사항 4-5).

입력 returns는 "로그수익률의 가중합" 근사치를 그대로 쓴다는 전제다
(포트폴리오 로그수익률의 엄밀한 정의가 아니라 자산별 로그수익률의
가중합으로 근사 — RL 논문/FinRL 등에서 흔히 쓰는 단순화. 리포트에는
이 근사를 명시할 것).

Compounding: env(`portfolio_env`)와 여기 모두 이 step return을 근사
로그수익률로 취급한다. 누적/MDD는 exp(cumsum) 경로를 쓰고, env의
portfolio_value도 np.exp(net_return)으로 복리한다. (1+r) 단순복리와
혼용하지 말 것.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _drawdown_series(returns: pd.Series) -> pd.Series:
    cum = np.exp(returns.cumsum())
    peak = cum.cummax()
    return (peak - cum) / peak


def compute_metrics(returns: pd.Series, benchmark: pd.Series | None = None, risk_free: float = 0.0) -> dict:
    returns = returns.dropna()
    n = len(returns)
    cumulative_return = float(np.expm1(returns.sum()))
    cagr = float((1 + cumulative_return) ** (TRADING_DAYS / n) - 1) if n > 0 else np.nan

    ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    var_95 = float(-np.percentile(returns, 5))
    tail = returns[returns <= np.percentile(returns, 5)]
    cvar_95 = float(-tail.mean()) if len(tail) > 0 else np.nan

    dd = _drawdown_series(returns)
    mdd = float(dd.max())

    mean_ann = returns.mean() * TRADING_DAYS
    sharpe = float((mean_ann - risk_free) / ann_vol) if ann_vol > 0 else np.nan

    downside = returns[returns < 0]
    downside_vol = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else np.nan
    sortino = float((mean_ann - risk_free) / downside_vol) if downside_vol and downside_vol > 0 else np.nan

    calmar = float(cagr / mdd) if mdd > 0 else np.nan

    alpha = beta = info_ratio = np.nan
    if benchmark is not None:
        bench = benchmark.reindex(returns.index).dropna()
        common = returns.index.intersection(bench.index)
        r, b = returns.loc[common], bench.loc[common]
        if len(common) > 2 and b.var() > 0:
            beta = float(np.cov(r, b)[0, 1] / np.var(b))
            alpha = float((r.mean() - beta * b.mean()) * TRADING_DAYS)
            active = r - b
            info_ratio = float(
                active.mean() * TRADING_DAYS / (active.std() * np.sqrt(TRADING_DAYS))
            ) if active.std() > 0 else np.nan

    return {
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "mdd": mdd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "alpha": alpha,
        "beta": beta,
        "information_ratio": info_ratio,
    }


def extend_after_safeguard(
    rets: np.ndarray | list[float],
    n_total: int,
    fee_rate: float,
    risk_free: float = 0.0,
) -> tuple[np.ndarray, int | None]:
    """Safe-Guard로 끊긴 테스트 롤아웃을 테스트 구간 끝까지 현금으로 채운다.

    이전 Walk-Forward는 MDD>15%로 에피소드가 끝나면 그 짧은 시계열을 1년
    성과로 채점했다(동일가중·MVO·SPY는 1년 전체). 그래서 2020년 초 테스트는
    DRL만 코로나 바닥(35거래일)에서 성적이 확정됐다.

    여기서는 Safe-Guard 발동을 "전량 청산 후 현금 보유"로 해석한다:
      - 발동 다음 날: 전량 매도 비용 1회(one-way turnover = 1.0) 차감
      - 이후: 무위험수익률(연율 risk_free → 일별 로그수익률)로 채움
    재진입은 하지 않는다(보수적 가정, 리포트에 명시).

    반환: (길이 n_total 시계열, 발동 인덱스 또는 None)
    """
    rets = np.asarray(rets, dtype=float)
    if len(rets) >= n_total:
        return rets[:n_total], None
    trigger_idx = len(rets) - 1
    n_pad = n_total - len(rets)
    daily_rf = float(np.log1p(risk_free) / TRADING_DAYS) if risk_free > -1 else 0.0
    pad = np.full(n_pad, daily_rf)
    pad[0] -= fee_rate
    return np.concatenate([rets, pad]), trigger_idx


def walk_forward_windows(
    dates: pd.DatetimeIndex,
    train_years: int = 4,
    test_years: int = 1,
    n_windows: int = 2,
) -> list[dict]:
    """Walk-Forward 윈도우(학습 구간, 테스트 구간) 목록을 만든다.

    예: Window1 = train[Y1~Y4] / test[Y5], Window2 = train[Y2~Y5] / test[Y6] ...
    윈도우가 데이터 범위를 벗어나면 그만큼 잘라서 반환한다.
    """
    start = dates.min()
    windows = []
    for i in range(n_windows):
        train_start = start + pd.DateOffset(years=i)
        train_end = train_start + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.DateOffset(days=1)
        if test_end > dates.max():
            break
        windows.append(
            {
                "train_start": train_start, "train_end": train_end,
                "test_start": test_start, "test_end": test_end,
            }
        )
    return windows
