"""금요일 데모용 end-to-end 스크립트.

더미 데이터로 전체 파이프라인(환경 -> 보상 3종 학습 -> 아웃오브샘플 평가 ->
MVO/동일가중 비교 -> ANOVA 2종 -> SHAP 해석)이 실제로 돌아가는 것을 보여준다.

주의: timesteps/자산수는 "짧은 시간에 파이프라인이 끝까지 도는 것"을
증명하는 데 맞춘 값이다. 실제 제출용 성능(10만 스텝 이상, 10자산 이상)은
train.py를 별도 인자로 오래 돌려서 얻어야 한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .backtest import compute_metrics
from .config import WINDOW_SIZE
from .env.portfolio_env import PortfolioEnv
from .mvo import rolling_mvo_backtest
from .pipeline import prepare_env_inputs
from .riskfree import fetch_risk_free_rate
from .shap_explain import build_feature_names, explain_decision
from .stats_tests import one_way_anova
from .train import _plot_learning_curve

OUT_DIR = Path(__file__).parent / "outputs" / "demo"


def rollout(model, env: PortfolioEnv):
    obs, _ = env.reset()
    returns, obs_history = [], []
    done = False
    while not done:
        obs_history.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        returns.append(info["net_return"])
        done = terminated or truncated
    return np.array(returns), np.array(obs_history)


def main(n_assets=6, window=WINDOW_SIZE, timesteps=8000, test_days=252, tickers=None, use_dummy=True, start="2019-01-01", end="2024-12-31", out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = tickers or [f"A{i}" for i in range(n_assets)]

    print(f"[demo] 데이터 준비 (use_dummy={use_dummy}): {tickers}")
    _, returns, rsi_df, macd_df = prepare_env_inputs(tickers, start=start, end=end, use_dummy=use_dummy)
    risk_free = fetch_risk_free_rate(start, end) if not use_dummy else 0.0
    print(f"[demo] risk_free_rate={risk_free:.4f} (연율화, BIL 기준)")

    split = len(returns) - test_days
    train_slice = slice(0, split)
    test_slice = slice(split - window, len(returns))

    reward_types = ["simple", "sharpe", "mdd_penalty"]
    trained = {}
    test_returns = {}

    for rt in reward_types:
        print(f"\n[demo] === reward={rt} 학습 시작 ({timesteps} steps) ===")
        train_env = PortfolioEnv(
            returns.iloc[train_slice], rsi_df.iloc[train_slice], macd_df.iloc[train_slice],
            window=window, reward_type=rt,
        )
        monitor_path = out_dir / f"monitor_{rt}"
        monitored_env = Monitor(train_env, filename=str(monitor_path))
        model = PPO("MlpPolicy", monitored_env, verbose=0, seed=0)
        model.learn(total_timesteps=timesteps)
        _plot_learning_curve(monitor_path, rt)

        test_env = PortfolioEnv(
            returns.iloc[test_slice], rsi_df.iloc[test_slice], macd_df.iloc[test_slice],
            window=window, reward_type=rt,
        )
        rets, obs_hist = rollout(model, test_env)
        trained[rt] = (model, test_env, obs_hist)
        test_returns[rt] = pd.Series(rets, name=rt)
        print(f"[demo] reward={rt} 아웃오브샘플 누적수익률={np.expm1(rets.sum()):.4f}")

    # --- 비교 기준: MVO, 동일가중 ---
    test_returns_df = returns.iloc[test_slice].iloc[window:]
    mvo_returns = rolling_mvo_backtest(returns.iloc[: split + test_days]).iloc[-len(test_returns_df):]
    equal_weight_returns = test_returns_df.mean(axis=1)

    all_series = {
        **{f"drl_{k}": v.reset_index(drop=True).set_axis(test_returns_df.index[: len(v)]) for k, v in test_returns.items()},
        "mvo": mvo_returns,
        "equal_weight": equal_weight_returns,
    }

    print("\n[demo] === 성과 지표 비교 (아웃오브샘플) ===")
    metrics_table = {}
    for name, series in all_series.items():
        metrics_table[name] = compute_metrics(series, benchmark=equal_weight_returns, risk_free=risk_free)
    metrics_df = pd.DataFrame(metrics_table).T
    print(metrics_df.round(4))
    metrics_df.to_csv(out_dir / "metrics_comparison.csv")

    # --- ANOVA 검증 1: 보상함수 3종 비교 ---
    reward_groups = {k: all_series[f"drl_{k}"] for k in reward_types}
    anova1 = one_way_anova(reward_groups)
    print(f"\n[demo] ANOVA(보상함수 3종): F={anova1['f_stat']:.3f}, p={anova1['p_value']:.4f}, eta^2={anova1['eta_squared']:.4f}")

    # --- ANOVA 검증 2: DRL(mdd_penalty) vs MVO vs 동일가중 ---
    strategy_groups = {
        "drl_mdd_penalty": all_series["drl_mdd_penalty"],
        "mvo": all_series["mvo"],
        "equal_weight": all_series["equal_weight"],
    }
    anova2 = one_way_anova(strategy_groups)
    print(f"[demo] ANOVA(DRL vs MVO vs 동일가중): F={anova2['f_stat']:.3f}, p={anova2['p_value']:.4f}, eta^2={anova2['eta_squared']:.4f}")

    # --- SHAP: mdd_penalty 모델의 마지막 시점 의사결정 설명 ---
    print("\n[demo] SHAP 해석 계산 중 (몇 분 걸릴 수 있음)...")
    model, test_env, obs_hist = trained["mdd_penalty"]
    feature_names = build_feature_names(tickers, window)
    top_asset_idx = int(np.argmax(test_env.weights))
    shap_paths = explain_decision(
        model, background_obs=obs_hist[:-1], target_obs=obs_hist[-1],
        asset_index=top_asset_idx, feature_names=feature_names,
        out_dir=out_dir, nsamples=100,
    )
    print(f"[demo] SHAP plots 저장: {shap_paths}")

    print(f"\n[demo] 모든 산출물 -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-assets", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=8000)
    parser.add_argument("--test-days", type=int, default=252)
    parser.add_argument("--tickers", type=str, default=None, help="쉼표로 구분된 실제 티커 목록 (예: SPY,QQQ,...)")
    parser.add_argument("--start", type=str, default="2019-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None
    main(
        n_assets=args.n_assets, timesteps=args.timesteps, test_days=args.test_days,
        tickers=tickers, use_dummy=tickers is None, start=args.start, end=args.end,
        out_dir=args.out_dir,
    )
