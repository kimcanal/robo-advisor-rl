"""보상 함수 3종 변형 (요구사항 4-3).

각 함수는 동일한 시그니처 reward_fn(step_return, history, **kwargs) -> float 를
따른다. history는 지금까지의 스텝별 포트폴리오 수익률 리스트(에피소드 내부).

변형별 설계 근거:
    simple      : reward = r_t
                  리스크를 전혀 고려하지 않는 베이스라인. 다른 두 변형과
                  비교했을 때 "리스크 조정을 하면 무엇이 달라지는가"를
                  보여주기 위한 대조군.
    sharpe      : reward = r_t / sigma_t (롤링 변동성으로 나눔)
                  변동성 대비 수익을 우대 -> 변동성이 큰 자산에 과도하게
                  쏠리는 것을 억제하는 효과를 기대.
    mdd_penalty : reward = r_t - lambda * MDD_t
                  누적 낙폭(drawdown)에 직접 페널티를 부과해 큰 손실 구간을
                  회피하도록 유도. lambda는 공격성/방어성 트레이드오프 노브.
"""
from __future__ import annotations

import numpy as np

from .config import MDD_LAMBDA_DEFAULT


def _rolling_vol(history: list[float], window: int = 20, eps: float = 1e-6) -> float:
    if len(history) < 2:
        return eps
    recent = history[-window:]
    vol = float(np.std(recent))
    return max(vol, eps)


def _current_drawdown(history: list[float]) -> float:
    if not history:
        return 0.0
    cum = np.cumprod([1 + r for r in history])
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    return float(dd[-1])


def reward_simple(step_return: float, history: list[float], **_) -> float:
    return step_return


def reward_sharpe(step_return: float, history: list[float], window: int = 20, **_) -> float:
    vol = _rolling_vol(history + [step_return], window=window)
    return step_return / vol


def reward_mdd_penalty(
    step_return: float,
    history: list[float],
    mdd_lambda: float = MDD_LAMBDA_DEFAULT,
    **_,
) -> float:
    dd = _current_drawdown(history + [step_return])
    return step_return - mdd_lambda * dd


REWARD_FUNCS = {
    "simple": reward_simple,
    "sharpe": reward_sharpe,
    "mdd_penalty": reward_mdd_penalty,
}


def get_reward_fn(name: str):
    if name not in REWARD_FUNCS:
        raise ValueError(f"unknown reward type '{name}', choose from {list(REWARD_FUNCS)}")
    return REWARD_FUNCS[name]
