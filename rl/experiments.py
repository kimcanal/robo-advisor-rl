"""보상함수/윈도우 하이퍼파라미터 실험 (요구사항 4-3, 4-2 설계 근거용).

미션 스펙이 명시적으로 요구하는 두 가지 실험을 자동화한다:
  1. mdd_penalty 보상의 lambda(0.5~5.0)에 따른 수익률-MDD 트레이드오프 곡선
  2. 관측 윈도우 N(20 vs 60)에 따른 학습 속도/성과 비교

각 실험은 run_demo.py와 동일한 train/test 분할 방식을 쓰되, 여러 설정을
반복 실행하고 결과를 표+그래프로 정리한다. 리포트 3장(환경설계)·4장
(보상함수)의 "선택 근거" 섹션에 그대로 쓸 수 있다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .backtest import compute_metrics
from .config import WINDOW_SIZE
from .env.portfolio_env import PortfolioEnv
from .pipeline import prepare_env_inputs
from .run_demo import rollout

OUT_DIR = Path(__file__).parent / "outputs" / "experiments"


def _split(returns, rsi_df, macd_df, window, test_days):
    split = len(returns) - test_days
    train_slice = slice(0, split)
    test_slice = slice(split - window, len(returns))
    return (
        (returns.iloc[train_slice], rsi_df.iloc[train_slice], macd_df.iloc[train_slice]),
        (returns.iloc[test_slice], rsi_df.iloc[test_slice], macd_df.iloc[test_slice]),
    )


def _train_and_eval(train_data, test_data, window, reward_type, reward_kwargs, timesteps, seed=0):
    train_env = PortfolioEnv(*train_data, window=window, reward_type=reward_type, reward_kwargs=reward_kwargs)
    model = PPO("MlpPolicy", train_env, verbose=0, seed=seed)
    model.learn(total_timesteps=timesteps)

    test_env = PortfolioEnv(*test_data, window=window, reward_type=reward_type, reward_kwargs=reward_kwargs)
    rets, _ = rollout(model, test_env)
    series = pd.Series(rets)
    metrics = compute_metrics(series)
    return metrics


def lambda_sweep(
    n_assets=10, window=WINDOW_SIZE, timesteps=15_000, test_days=252,
    lambdas=(0.5, 1.0, 2.0, 5.0),
):
    """변형3 보상(reward = r_t - lambda*MDD_t)의 lambda를 바꿔가며
    수익률-MDD 트레이드오프를 확인한다.

    기대 방향: lambda가 클수록(방어적) MDD는 줄고 수익률도 함께 낮아져야
    하고, lambda가 작을수록(공격적) 그 반대가 되어야 한다. 이 단조적
    경향이 실제로 나오는지가 "보상 설계가 의도대로 작동하는지"의 근거가 된다.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = [f"A{i}" for i in range(n_assets)]
    _, returns, rsi_df, macd_df = prepare_env_inputs(tickers, use_dummy=True, verbose=False)
    train_data, test_data = _split(returns, rsi_df, macd_df, window, test_days)

    rows = []
    for lam in lambdas:
        print(f"[lambda_sweep] lambda={lam} 학습 중...")
        metrics = _train_and_eval(
            train_data, test_data, window, "mdd_penalty", {"mdd_lambda": lam}, timesteps,
        )
        rows.append({"lambda": lam, **metrics})
        print(f"[lambda_sweep] lambda={lam} -> return={metrics['cumulative_return']:.4f}, mdd={metrics['mdd']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "lambda_sweep.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(df["lambda"], df["cumulative_return"], "o-", color="tab:blue", label="cumulative return")
    ax1.set_xlabel("lambda (MDD penalty strength)")
    ax1.set_ylabel("cumulative return", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df["lambda"], df["mdd"], "s--", color="tab:red", label="MDD")
    ax2.set_ylabel("MDD", color="tab:red")
    plt.title("Reward variant 3: return vs MDD trade-off across lambda")
    fig.tight_layout()
    plot_path = OUT_DIR / "lambda_tradeoff.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"[lambda_sweep] saved -> {OUT_DIR / 'lambda_sweep.csv'}, {plot_path}")
    return df


def window_sweep(n_assets=10, windows=(20, 60), timesteps=15_000, test_days=252):
    """관측 윈도우 N=20 vs 60 비교. N이 커지면 관측 차원이 늘어 학습이
    느려지는지, 성과가 달라지는지를 확인해 "왜 이 N을 골랐는지" 근거로 쓴다.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = [f"A{i}" for i in range(n_assets)]

    rows = []
    for w in windows:
        _, returns, rsi_df, macd_df = prepare_env_inputs(tickers, use_dummy=True, verbose=False)
        train_data, test_data = _split(returns, rsi_df, macd_df, w, test_days)
        print(f"[window_sweep] window={w} 학습 중...")
        metrics = _train_and_eval(train_data, test_data, w, "mdd_penalty", {"mdd_lambda": 1.0}, timesteps)
        obs_dim = w * n_assets + 3 * n_assets
        rows.append({"window": w, "obs_dim": obs_dim, **metrics})
        print(f"[window_sweep] window={w} -> return={metrics['cumulative_return']:.4f}, mdd={metrics['mdd']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "window_sweep.csv", index=False)
    print(f"[window_sweep] saved -> {OUT_DIR / 'window_sweep.csv'}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["lambda", "window", "both"], default="both")
    parser.add_argument("--timesteps", type=int, default=15_000)
    args = parser.parse_args()

    if args.which in ("lambda", "both"):
        lambda_sweep(timesteps=args.timesteps)
    if args.which in ("window", "both"):
        window_sweep(timesteps=args.timesteps)
