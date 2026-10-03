"""학습 곡선(요구사항 4-4)과 변형3 lambda 곡선(요구사항 4-3).

학습 곡선:
    python -m rl.curves learning --monitor-dir artifacts/drl/monitor --out artifacts/drl
    - walk_forward --save-dir 가 남긴 {run_tag}.monitor.csv (에피소드 보상 r, 길이 l)
    - 보상 3종은 스케일이 달라 같은 축에 그리지 않는다(패널 분리)
    - 에피소드 길이도 그린다: Safe-Guard(MDD 15%)로 일찍 끝나는 에피소드가 줄어드는지
    - 수렴 지표: 마지막 20% 에피소드 평균 vs 직전 20% 평균의 차이 (숫자만 보고, 판정은 사람이)

lambda 곡선:
    python -m rl.curves lambda --universe 16 --start 2015-01-01 --window-index 7 \
        --lambdas 0.5,1,2,3,5 --seeds 0,1,2 --timesteps 120000 --out artifacts/drl
    - walk_forward.run 을 그대로 써서 데이터·평가(절단 수정 포함)가 본 실험과 같다
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TAG_RE = re.compile(r"w(?P<window>\d+)_(?P<reward>simple|sharpe|mdd_penalty)_s(?P<seed>\d+)")


def load_monitors(monitor_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(Path(monitor_dir).glob("*.monitor.csv")):
        m = TAG_RE.search(f.name)
        if not m:
            continue
        df = pd.read_csv(f, skiprows=1)
        if df.empty:
            continue
        df["episode"] = np.arange(1, len(df) + 1)
        df["timesteps"] = df["l"].cumsum()
        df["window"] = int(m["window"])
        df["reward_type"] = m["reward"]
        df["seed"] = int(m["seed"])
        frames.append(df[["window", "reward_type", "seed", "episode", "timesteps", "r", "l"]])
    if not frames:
        return pd.DataFrame(columns=["window", "reward_type", "seed", "episode", "timesteps", "r", "l"])
    return pd.concat(frames, ignore_index=True)


def convergence_stats(df: pd.DataFrame) -> list[dict]:
    out = []
    for (w, rt, sd), g in df.groupby(["window", "reward_type", "seed"]):
        n = len(g)
        k = max(1, n // 5)
        last, prev = g.r.iloc[-k:], g.r.iloc[-2 * k:-k] if n >= 2 * k else g.r.iloc[:0]
        out.append({
            "window": int(w), "reward_type": rt, "seed": int(sd), "episodes": int(n),
            "timesteps": int(g.timesteps.iloc[-1]),
            "last20_mean_reward": float(last.mean()),
            "prev20_mean_reward": float(prev.mean()) if len(prev) else None,
            "last20_minus_prev20": float(last.mean() - prev.mean()) if len(prev) else None,
            "first20_mean_len": float(g.l.iloc[:k].mean()),
            "last20_mean_len": float(g.l.iloc[-k:].mean()),
        })
    return out


def downsample_for_api(df: pd.DataFrame, max_points: int = 200, roll: int = 10) -> list[dict]:
    rows = []
    for (w, rt, sd), g in df.groupby(["window", "reward_type", "seed"]):
        g = g.copy()
        g["r_roll"] = g.r.rolling(roll, min_periods=1).mean()
        g["l_roll"] = g.l.rolling(roll, min_periods=1).mean()
        idx = np.unique(np.linspace(0, len(g) - 1, num=min(max_points, len(g))).astype(int))
        for _, r in g.iloc[idx].iterrows():
            rows.append({"window": int(w), "reward_type": rt, "seed": int(sd),
                         "timesteps": int(r.timesteps), "episode_reward": float(r.r),
                         "reward_rolling": float(r.r_roll), "episode_len_rolling": float(r.l_roll)})
    return rows


def plot_learning_curves(df: pd.DataFrame, out_png: Path, roll: int = 10) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rewards = [r for r in ("simple", "sharpe", "mdd_penalty") if r in set(df.reward_type)]
    fig, axes = plt.subplots(2, len(rewards), figsize=(4.2 * len(rewards), 6.5), squeeze=False)
    for j, rt in enumerate(rewards):
        g = df[df.reward_type == rt]
        for (w, sd), gg in g.groupby(["window", "seed"]):
            lab = f"w{w} s{sd}"
            axes[0, j].plot(gg.timesteps, gg.r.rolling(roll, min_periods=1).mean(), label=lab, lw=1)
            axes[1, j].plot(gg.timesteps, gg.l.rolling(roll, min_periods=1).mean(), label=lab, lw=1)
        axes[0, j].set_title(f"reward={rt}")
        axes[0, j].set_ylabel(f"episode reward (rolling {roll})")
        axes[1, j].set_ylabel("episode length (days)")
        axes[1, j].set_xlabel("timesteps")
        axes[0, j].legend(fontsize=7)
    fig.suptitle("PPO learning curves (Monitor, raw env reward; scales differ by reward)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def learning(monitor_dir: Path, out: Path) -> dict:
    df = load_monitors(monitor_dir)
    if df.empty:
        raise FileNotFoundError(f"no *.monitor.csv with run tags under {monitor_dir}")
    png = plot_learning_curves(df, out / "learning_curves.png")
    payload = {"convergence": convergence_stats(df), "points": downsample_for_api(df),
               "plot": png.name, "source": str(monitor_dir)}
    (out / "learning_curves.json").write_text(json.dumps(payload, indent=2))
    print(f"[curves] learning curves: {df.groupby(['reward_type','seed']).ngroups} runs → {png}")
    return payload


def lambda_sweep(lambdas, seeds, timesteps, window_index, universe, start, end, out: Path) -> pd.DataFrame:
    from .walk_forward import REAL_TICKERS, TICKERS_16, run

    rows = []
    for lam in lambdas:
        df, _ = run(
            tickers=TICKERS_16 if universe == "16" else REAL_TICKERS, use_dummy=False,
            start=start, end=end, timesteps=timesteps, seeds=tuple(seeds),
            reward_types=("mdd_penalty",), n_windows=window_index,
            only_windows=(window_index,), mdd_lambda=lam,
            out_dir=out / "lambda_runs" / f"lambda_{lam}",
        )
        d = df[df.strategy == "drl_mdd_penalty"].copy()
        d["lambda"] = lam
        rows.append(d)
        part = pd.concat(rows, ignore_index=True)
        part.to_csv(out / "lambda_sweep.csv", index=False)
    res = pd.concat(rows, ignore_index=True)
    plot_lambda(res, out / "lambda_tradeoff.png")
    return res


def plot_lambda(res: pd.DataFrame, out_png: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = res.groupby("lambda").agg(cum_mean=("cumulative_return", "mean"),
                                    cum_min=("cumulative_return", "min"),
                                    cum_max=("cumulative_return", "max"),
                                    mdd_mean=("mdd", "mean"), mdd_min=("mdd", "min"),
                                    mdd_max=("mdd", "max")).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(agg["lambda"], agg.cum_mean, "o-")
    ax[0].fill_between(agg["lambda"], agg.cum_min, agg.cum_max, alpha=0.2)
    ax[0].set_xlabel("lambda")
    ax[0].set_ylabel("test cumulative return (seed mean, band=min..max)")
    ax[1].plot(agg.mdd_mean, agg.cum_mean, "o-")
    for _, r in agg.iterrows():
        ax[1].annotate(f"λ={r['lambda']:g}", (r.mdd_mean, r.cum_mean), fontsize=8)
    ax[1].set_xlabel("MDD (seed mean)")
    ax[1].set_ylabel("cumulative return (seed mean)")
    ax[1].set_title("return–MDD trade-off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("learning")
    a1.add_argument("--monitor-dir", required=True)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("lambda")
    a2.add_argument("--lambdas", default="0.5,1,2,3,5")
    a2.add_argument("--seeds", default="0,1,2")
    a2.add_argument("--timesteps", type=int, default=120_000)
    a2.add_argument("--window-index", type=int, default=7)
    a2.add_argument("--universe", choices=["10", "16"], default="16")
    a2.add_argument("--start", default="2015-01-01")
    a2.add_argument("--end", default="2026-09-29")
    a2.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "learning":
        learning(Path(a.monitor_dir), Path(a.out))
    else:
        lambda_sweep([float(x) for x in a.lambdas.split(",")], [int(x) for x in a.seeds.split(",")],
                     a.timesteps, a.window_index, a.universe, a.start, a.end, Path(a.out))
