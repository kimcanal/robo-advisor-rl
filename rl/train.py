"""PPO 학습 스크립트 (요구사항 4-4).

사용 예:
    python -m rl.train --reward mdd_penalty --timesteps 20000

기본은 더미 데이터 10자산. 데이터팀 CSV가 준비되면 --use-dummy 0 만
붙이면 동일 코드로 실제 데이터 학습이 된다 (인터페이스는 loader.py 계약 참고).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .config import MDD_LAMBDA_DEFAULT, WINDOW_SIZE
from .env.portfolio_env import PortfolioEnv
from .pipeline import prepare_env_inputs

DEFAULT_TICKERS = [f"A{i}" for i in range(10)]
OUT_DIR = Path(__file__).parent / "outputs"


def build_env(tickers, start, end, use_dummy, reward_type, window):
    _, returns, rsi_df, macd_df = prepare_env_inputs(tickers, start, end, use_dummy=use_dummy)
    reward_kwargs = {"mdd_lambda": MDD_LAMBDA_DEFAULT} if reward_type == "mdd_penalty" else {}
    env = PortfolioEnv(
        returns, rsi_df, macd_df,
        window=window, reward_type=reward_type, reward_kwargs=reward_kwargs,
    )
    return env


def train(
    reward_type: str = "mdd_penalty",
    timesteps: int = 20_000,
    tickers=None,
    start="2019-01-01",
    end="2024-12-31",
    use_dummy: bool = True,
    window: int = WINDOW_SIZE,
    seed: int = 0,
):
    tickers = tickers or DEFAULT_TICKERS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "models").mkdir(exist_ok=True)
    (OUT_DIR / "plots").mkdir(exist_ok=True)
    (OUT_DIR / "logs").mkdir(exist_ok=True)

    env = build_env(tickers, start, end, use_dummy, reward_type, window)
    monitor_path = OUT_DIR / "logs" / f"monitor_{reward_type}"
    env = Monitor(env, filename=str(monitor_path))

    model = PPO("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=timesteps)

    model_path = OUT_DIR / "models" / f"ppo_{reward_type}.zip"
    model.save(model_path)

    plot_path = _plot_learning_curve(monitor_path, reward_type)
    print(f"[train] saved model -> {model_path}")
    print(f"[train] saved learning curve -> {plot_path}")
    return model, str(model_path), str(plot_path)


def _plot_learning_curve(monitor_path: Path, reward_type: str) -> Path:
    csv_path = Path(str(monitor_path) + ".monitor.csv")
    df = pd.read_csv(csv_path, skiprows=1)
    df["cum_steps"] = df["l"].cumsum()

    plt.figure(figsize=(8, 4))
    plt.plot(df["cum_steps"], df["r"], alpha=0.4, label="episode reward")
    if len(df) >= 5:
        plt.plot(df["cum_steps"], df["r"].rolling(5, min_periods=1).mean(), label="rolling mean (5)")
    plt.xlabel("timesteps")
    plt.ylabel("episode reward")
    plt.title(f"PPO learning curve — reward={reward_type}")
    plt.legend()
    plt.tight_layout()
    out_path = OUT_DIR / "plots" / f"learning_curve_{reward_type}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", default="mdd_penalty", choices=["simple", "sharpe", "mdd_penalty"])
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--use-dummy", type=int, default=1)
    args = parser.parse_args()

    train(reward_type=args.reward, timesteps=args.timesteps, use_dummy=bool(args.use_dummy))
