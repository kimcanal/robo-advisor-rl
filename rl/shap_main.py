"""본학습 PPO 모델 SHAP (요구사항 4-7) — Summary 1개 + Force 여러 개 + API용 JSON.

    python -m rl.shap_main --artifacts artifacts/drl [--run-tag w7_mdd_penalty_s0]

왜 그룹 SHAP인가
    관측은 30일×16종목 수익률(480) + 비중·RSI·MACD(각 16) + 리스크(1) = 529차원이다.
    개별 피처 Kernel SHAP은 샘플 수 대비 차원이 커서 추정이 불안정하고 읽을 수도
    없다. 그래서 종목별로 {30일 수익률창, 현재비중, RSI, MACD} 4개 묶음 + 리스크
    1개 = 65개 그룹을 하나의 플레이어로 두고 Shapley 값을 구한다(묶음 단위 마스킹,
    shap 레거시 DenseData groups). 그룹 SHAP의 합은 여전히
    f(x) - E[f(background)] 와 같다 (가법성 유지).

설명 대상(스칼라)
    정책 출력(비중 벡터) 중 한 자산의 비중 = softmax(clip(PPO action))[asset].
    입력은 raw 관측이고, 함수 안에서 VecNormalize 통계로 정규화한다(학습과 동일 경로).

background
    학습 구간을 정책으로 굴리며 모은 raw 관측에서 무작위 표본(기본 30개).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .env.portfolio_env import PortfolioEnv
from .pipeline import prepare_env_inputs
from .serving import PolicyRun, action_to_weights, artifact_dir, load_run, serving_config


# ------------------------------------------------------------------- groups
def feature_groups(tickers: list[str], window: int) -> tuple[list[str], list[np.ndarray]]:
    n = len(tickers)
    names, groups = [], []
    for j, t in enumerate(tickers):
        names.append(f"{t}:ret{window}d")
        groups.append(np.array([k * n + j for k in range(window)]))
    base = window * n
    for label, off in (("weight", base), ("rsi", base + n), ("macd", base + 2 * n)):
        for j, t in enumerate(tickers):
            names.append(f"{t}:{label}")
            groups.append(np.array([off + j]))
    names.append("portfolio_risk")
    groups.append(np.array([base + 3 * n]))
    return names, groups


def group_display_values(obs: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """그룹별 대표값 (수익률창은 30일 평균 로그수익률, 나머지는 단일값)."""
    obs = np.atleast_2d(obs)
    return np.stack([obs[:, g].mean(axis=1) for g in groups], axis=1)


# ------------------------------------------------------------------ rollout
def policy_rollout(run: PolicyRun, returns, rsi_df, macd_df) -> pd.DataFrame:
    """PortfolioEnv를 정책으로 굴려 raw 관측·비중·수익률을 모은다 (Safe-Guard 시 종료)."""
    env = PortfolioEnv(returns, rsi_df, macd_df, window=run.window,
                       reward_type=run.meta["reward_type"],
                       reward_kwargs=run.meta.get("reward_kwargs") or {})
    model = run.load_model()
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        t = env.t
        action, _ = model.predict(run.norm(obs)[None, :], deterministic=True)
        weights = action_to_weights(np.asarray(action)[0])
        rows.append({"as_of": returns.index[t - 1], "obs": obs.copy(), "weights": weights})
        obs, _r, term, trunc, info = env.step(np.asarray(action)[0])
        rows[-1]["net_return"] = info["net_return"]
        rows[-1]["safe_guard"] = bool(info["safe_guard_triggered"])
        done = term or trunc
    return pd.DataFrame(rows)


def _scalar_fn(run: PolicyRun, asset_index: int):
    model = run.load_model()

    def f(raw_batch: np.ndarray) -> np.ndarray:
        raw_batch = np.atleast_2d(raw_batch)
        normed = np.stack([run.norm(r) for r in raw_batch])
        actions, _ = model.predict(normed, deterministic=True)
        return action_to_weights(np.atleast_2d(actions))[:, asset_index]

    return f


# --------------------------------------------------------------------- shap
def grouped_kernel_shap(f, background: np.ndarray, X: np.ndarray, names, groups,
                        nsamples: int | str = "auto"):
    import shap
    from shap.utils._legacy import DenseData

    explainer = shap.KernelExplainer(f, DenseData(background, names, groups))
    sv = np.asarray(explainer.shap_values(X, nsamples=nsamples, silent=True))
    sv = sv.reshape(len(X), len(names))
    return float(np.asarray(explainer.expected_value).ravel()[0]), sv


def explain_run(run: PolicyRun, *, start: str, end: str, use_dummy: bool = False,
                out_dir: Path, n_background: int = 30, n_summary: int = 40,
                nsamples: int | str = "auto", seed: int = 0) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, tickers, window = run.meta, run.tickers, run.window
    _, returns, rsi_df, macd_df = prepare_env_inputs(tickers, start=start, end=end,
                                                     use_dummy=use_dummy, verbose=False)
    train = slice(meta["train_start"], meta["train_end"])
    test_pos0 = returns.index.get_indexer([pd.Timestamp(meta["test_start"])], method="bfill")[0]
    test_pos1 = returns.index.get_indexer([pd.Timestamp(meta["test_end"])], method="ffill")[0]
    test_idx = returns.index[max(0, test_pos0 - window): test_pos1 + 1]

    tr = policy_rollout(run, returns.loc[train], rsi_df.loc[train], macd_df.loc[train])
    te = policy_rollout(run, returns.loc[test_idx], rsi_df.loc[test_idx], macd_df.loc[test_idx])
    bg = np.stack(tr.obs.to_numpy())[rng.choice(len(tr), size=min(n_background, len(tr)), replace=False)]
    test_obs = np.stack(te.obs.to_numpy())
    test_w = np.stack(te.weights.to_numpy())
    names, groups = feature_groups(tickers, window)

    # Summary: 테스트 구간 평균 비중이 가장 큰 자산
    focus = int(np.argmax(test_w.mean(axis=0)))
    pick = np.linspace(0, len(te) - 1, num=min(n_summary, len(te))).astype(int)
    ev, sv = grouped_kernel_shap(_scalar_fn(run, focus), bg, test_obs[pick], names, groups, nsamples)
    disp = group_display_values(test_obs[pick], groups)
    plt.figure()
    shap.summary_plot(sv, disp, feature_names=names, show=False, max_display=20)
    plt.title(f"SHAP summary — weight of {tickers[focus]} ({meta['run_tag']}, test {meta['test_start'][:7]}~)")
    plt.tight_layout()
    summary_png = out_dir / "shap_summary.png"
    plt.savefig(summary_png, dpi=140, bbox_inches="tight")
    plt.close()
    mean_abs = sorted(
        ({"feature": n, "mean_abs_shap": float(v)} for n, v in zip(names, np.abs(sv).mean(axis=0))),
        key=lambda r: -r["mean_abs_shap"],
    )

    # Force: 마지막 결정, 최악 일간 수익 직전 결정, Safe-Guard 발동 결정(있으면)
    targets = {"last_decision": len(te) - 1, "worst_day": int(np.argmin(te.net_return.to_numpy()))}
    sg = np.flatnonzero(te.safe_guard.to_numpy())
    if len(sg):
        # 롤아웃은 Safe-Guard 날 끝나므로 마지막 결정 = 발동 결정. 라벨을 바꿔 중복을 없앤다.
        if int(sg[0]) == targets["last_decision"]:
            targets.pop("last_decision")
        targets["safe_guard"] = int(sg[0])
    decisions = []
    for label, i in targets.items():
        asset = int(np.argmax(test_w[i]))
        ev_i, sv_i = grouped_kernel_shap(_scalar_fn(run, asset), bg, test_obs[i:i + 1], names, groups, nsamples)
        disp_i = group_display_values(test_obs[i], groups)[0]
        plt.figure()
        shap.force_plot(ev_i, sv_i[0], np.round(disp_i, 4), feature_names=names,
                        matplotlib=True, show=False, text_rotation=20)
        png = out_dir / f"shap_force_{label}.png"
        plt.savefig(png, dpi=140, bbox_inches="tight")
        plt.close("all")
        order = np.argsort(-np.abs(sv_i[0]))
        decisions.append({
            "label": label,
            "as_of": str(pd.Timestamp(te.as_of.iloc[i]).date()),
            "asset": tickers[asset],
            "asset_index": asset,
            "base_value": ev_i,
            "prediction": float(test_w[i, asset]),
            "contributions": [
                {"feature": names[k], "shap": float(sv_i[0, k]), "value": float(disp_i[k])}
                for k in order
            ],
            "force_plot": png.name,
        })

    result = {
        "run_tag": meta["run_tag"],
        "reward_type": meta["reward_type"],
        "train": [meta["train_start"], meta["train_end"]],
        "test": [meta["test_start"], meta["test_end"]],
        "method": "KernelExplainer, grouped (per-ticker returns-window/weight/rsi/macd + risk)",
        "n_background": int(len(bg)),
        "nsamples": nsamples,
        "summary": {"asset": tickers[focus], "n_explained": int(len(pick)),
                    "expected_value": ev, "mean_abs_shap": mean_abs, "plot": summary_png.name},
        "decisions": decisions,
        "test_policy_days": int(len(te)),
        "note": "portfolio_risk is 0 for all dates (no news risk tags in training).",
    }
    (out_dir / "shap_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(artifact_dir()))
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-09-29")
    ap.add_argument("--nsamples", default="auto")
    ap.add_argument("--n-summary", type=int, default=40)
    a = ap.parse_args()
    root = Path(a.artifacts)
    tag = a.run_tag or (serving_config(root) or {}).get("run_tags", [None])[0]
    if not tag:
        raise SystemExit("no run tag (pass --run-tag or export serving first)")
    ns = a.nsamples if a.nsamples == "auto" else int(a.nsamples)
    res = explain_run(load_run(root / "models" / tag), start=a.start, end=a.end,
                      out_dir=root / "shap", nsamples=ns, n_summary=a.n_summary)
    print(f"[shap_main] {res['run_tag']}: summary asset={res['summary']['asset']}, "
          f"decisions={[d['label'] for d in res['decisions']]} → {root/'shap'}")
