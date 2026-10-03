"""Walk-Forward 결과 CSV → ANOVA 3종, Safe-Guard 절단 점검, 성과 목표 점검.

학습을 다시 돌리지 않고, 이미 저장된 `walk_forward_results*.csv`(윈도우×전략×시드
12지표)만으로 계산할 수 있는 것을 전부 계산한다. 숫자는 CSV에서만 나온다.

사용:
    python -m rl.analysis.wf_analysis \
        --results docs/results/walk_forward_results_16assets_2015.csv \
        [--daily path/to/walk_forward_daily.csv] \
        --out docs/results

출력:
    {out}/anova_results.json     검증1·2·3 + 보조 분석 (API GET /anova/results)
    {out}/wf_summary.json        윈도우 요약, 절단 행, 성과 목표 점검 (API GET /backtest)
    {out}/figures/*.png          리포트용 그림

절단(truncation) 판별:
    compute_metrics는 CAGR = (1+cum)^(252/n) - 1 이므로
    n = 252 * ln(1+cum) / ln(1+CAGR). 같은 윈도우 동일가중의 n보다 짧으면
    Safe-Guard 등으로 테스트 롤아웃이 1년을 채우지 못한 행이다.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from ..stats_tests import ALPHA, one_way_anova, tukey_records, two_way_anova

DRL_REWARDS = ("drl_simple", "drl_sharpe", "drl_mdd_penalty")
METRIC_COLS = [
    "cumulative_return", "cagr", "ann_vol", "var_95", "cvar_95", "mdd",
    "sharpe", "sortino", "calmar", "alpha", "beta", "information_ratio",
]
# 성과 목표 (Notion 7. 제약사항 - 성능 기준)
TARGET_SHARPE_VS_EW = 0.2
TARGET_CUMRET_VS_BENCH_PP = 10.0
# 동일가중 대비 거래일이 이 비율 미만이면 절단으로 본다
TRUNCATION_RATIO = 0.97


def implied_test_days(cum: float, cagr: float) -> float:
    """compute_metrics의 CAGR 정의를 역산해 실제 평가 거래일 수를 구한다."""
    if cum is None or cagr is None or not np.isfinite(cum) or not np.isfinite(cagr):
        return float("nan")
    a, b = math.log1p(cum), math.log1p(cagr)
    if abs(b) < 1e-12:
        return float("nan")
    return 252.0 * a / b


def add_test_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_days"] = [implied_test_days(c, g) for c, g in zip(df["cumulative_return"], df["cagr"])]
    ew_days = df[df.strategy == "equal_weight"].set_index("window")["n_days"]
    df["ew_days"] = df["window"].map(ew_days)
    df["truncated"] = df["n_days"] < TRUNCATION_RATIO * df["ew_days"]
    return df


def _clean(obj):
    """JSON 직렬화용: numpy/NaN → 파이썬 기본형/None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items() if k != "tukey_hsd"}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _anova_table_records(table: pd.DataFrame) -> list[dict]:
    ss_total = table["sum_sq"].sum()
    out = []
    for term, row in table.iterrows():
        out.append(
            {
                "term": str(term),
                "sum_sq": float(row["sum_sq"]),
                "df": float(row["df"]),
                "F": None if pd.isna(row.get("F")) else float(row["F"]),
                "p_value": None if pd.isna(row.get("PR(>F)")) else float(row["PR(>F)"]),
                "eta_squared": float(row["sum_sq"] / ss_total) if ss_total > 0 else None,
            }
        )
    return out


def validation1(df: pd.DataFrame, metric: str = "cumulative_return") -> dict:
    """검증1: 보상 함수 3종 One-way ANOVA (윈도우·시드 풀링)."""
    groups = {r: df.loc[df.strategy == r, metric] for r in DRL_REWARDS}
    res = one_way_anova(groups)
    res["metric"] = metric
    res["design"] = "one-way, 7 windows x 3 seeds pooled per reward"
    return res


def validation1_blocked(df: pd.DataFrame, metric: str = "cumulative_return") -> dict:
    """보조: 보상 × 윈도우 Two-way (윈도우를 블록으로 통제). 명세 외 보조 분석."""
    drl = df[df.strategy.isin(DRL_REWARDS)].copy()
    model = ols(f"{metric} ~ C(strategy) + C(window)", data=drl).fit()
    table = sm.stats.anova_lm(model, typ=2)
    out = {
        "metric": metric,
        "design": "two-way additive (reward + window block), supplementary",
        "table": _anova_table_records(table),
    }
    p_strat = float(table.loc["C(strategy)", "PR(>F)"])
    out["reward_p_value"] = p_strat
    out["tukey_required"] = bool(p_strat < ALPHA)
    if p_strat < ALPHA:
        # 윈도우 평균을 빼고(블록 효과 제거) Tukey. 자유도는 근사 — 리포트에 명시.
        centered = drl[metric] - drl.groupby("window")[metric].transform("mean")
        _, recs = tukey_records(centered.to_numpy(), drl["strategy"].to_numpy())
        out["tukey_records_window_centered"] = recs
        out["tukey_note"] = (
            "Approximate post-hoc on window-centered values; df not adjusted for the "
            "window block. Treat as descriptive."
        )
    return out


def validation2(df: pd.DataFrame, drl_name: str = "drl_mdd_penalty",
                metric: str = "cumulative_return") -> dict:
    """검증2: DRL vs MVO vs 동일가중 One-way (walk_forward.py와 같은 그룹 구성)."""
    groups = {k: df.loc[df.strategy == k, metric] for k in (drl_name, "mvo", "equal_weight")}
    res = one_way_anova(groups)
    res["metric"] = metric
    res["drl_group"] = drl_name
    return res


def validation3(daily: pd.DataFrame | None) -> dict:
    """검증3: 전략 × 시장국면 Two-way. 일별 CSV가 있을 때만 계산."""
    if daily is None:
        return {
            "status": "not_reproducible_here",
            "reason": (
                "walk_forward_daily.csv is not committed; two-way ANOVA needs daily returns "
                "with regime labels. Re-run rl.walk_forward and pass --daily."
            ),
            "reported_in_pr7": {
                "strategy_p_value": 0.013,
                "strategy_eta_squared_upper_bound": 0.001,
                "source": "docs/rl_status_2026-10-04.md (Colab console output, F not recorded)",
            },
        }
    res = two_way_anova(daily, value_col="daily_return", factor1="strategy", factor2="regime")
    table = res["anova_table"]
    out = {"status": "computed", "table": _anova_table_records(table)}
    p_strat = float(table.loc["C(strategy)", "PR(>F)"])
    out["tukey_required"] = bool(p_strat < ALPHA)
    if p_strat < ALPHA:
        _, recs = tukey_records(daily["daily_return"].to_numpy(), daily["strategy"].to_numpy())
        out["tukey_records_strategy"] = recs
    return out


def window_summary(df: pd.DataFrame) -> list[dict]:
    rows = []
    for w, g in df.groupby("window"):
        drl = g[g.strategy.str.startswith("drl_")]
        base = g.set_index("strategy")
        row = {
            "window": int(w),
            "drl_cumret_min": float(drl.cumulative_return.min()),
            "drl_cumret_max": float(drl.cumulative_return.max()),
            "drl_sharpe_min": float(drl.sharpe.min()),
            "drl_sharpe_max": float(drl.sharpe.max()),
            "drl_truncated_rows": int(drl.truncated.sum()),
            "drl_min_days": float(drl.n_days.min()),
            "ew_days": float(g.ew_days.iloc[0]),
        }
        for b in ("equal_weight", "mvo", "spy", "kospi"):
            if b in base.index:
                row[f"{b}_cumret"] = float(base.loc[b, "cumulative_return"])
                row[f"{b}_sharpe"] = float(base.loc[b, "sharpe"])
        rows.append(row)
    return rows


def target_check(df: pd.DataFrame) -> dict:
    """성과 목표: 샤프 (DRL − 동일가중) ≥ +0.2, 누적 (DRL − SPY) ≥ +10%p."""
    rows = []
    for _, r in df[df.strategy.str.startswith("drl_")].iterrows():
        g = df[df.window == r.window].set_index("strategy")
        d_sharpe = r.sharpe - g.loc["equal_weight", "sharpe"]
        d_cum_pp = (r.cumulative_return - g.loc["spy", "cumulative_return"]) * 100
        rows.append(
            {
                "window": int(r.window), "strategy": r.strategy, "seed": r.seed,
                "sharpe_minus_ew": float(d_sharpe), "cumret_minus_spy_pp": float(d_cum_pp),
                "meets_sharpe": bool(d_sharpe >= TARGET_SHARPE_VS_EW),
                "meets_cumret": bool(d_cum_pp >= TARGET_CUMRET_VS_BENCH_PP),
                "truncated": bool(r.truncated),
            }
        )
    t = pd.DataFrame(rows)
    return {
        "n_drl_runs": int(len(t)),
        "meets_sharpe": int(t.meets_sharpe.sum()),
        "meets_cumret": int(t.meets_cumret.sum()),
        "meets_both": int((t.meets_sharpe & t.meets_cumret).sum()),
        "mean_sharpe_minus_ew": float(t.sharpe_minus_ew.mean()),
        "mean_cumret_minus_spy_pp": float(t.cumret_minus_spy_pp.mean()),
        "rows": rows,
    }


def make_figures(df: pd.DataFrame, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    windows = sorted(df.window.unique())
    colors = {"drl_simple": "tab:blue", "drl_sharpe": "tab:orange", "drl_mdd_penalty": "tab:green"}
    for metric, fname, ylabel in (
        ("cumulative_return", "wf_cumret_by_window.png", "cumulative return (test year)"),
        ("sharpe", "wf_sharpe_by_window.png", "Sharpe ratio"),
    ):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, rt in enumerate(DRL_REWARDS):
            g = df[df.strategy == rt]
            x = g.window + (i - 1) * 0.18
            trunc = g.truncated
            ax.scatter(x[~trunc], g[metric][~trunc], color=colors[rt], label=rt, s=22)
            ax.scatter(x[trunc], g[metric][trunc], color=colors[rt], marker="x", s=40)
        for b, style in (("equal_weight", "k--"), ("mvo", "m:"), ("spy", "r-.")):
            g = df[df.strategy == b].sort_values("window")
            ax.plot(g.window, g[metric], style, label=b, linewidth=1.2)
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_xticks(windows)
        ax.set_xlabel("walk-forward window (x = DRL rollout truncated by Safe-Guard)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(str(p))

    fig, ax = plt.subplots(figsize=(9, 3.5))
    drl = df[df.strategy.str.startswith("drl_")]
    ax.scatter(drl.window, drl.n_days, c=drl.truncated.map({True: "tab:red", False: "tab:blue"}), s=18)
    ew = df[df.strategy == "equal_weight"].sort_values("window")
    ax.plot(ew.window, ew.n_days, "k--", label="equal_weight (full year)")
    ax.set_xticks(windows)
    ax.set_ylabel("evaluated trading days")
    ax.set_xlabel("window (red = truncated DRL rollout)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "wf_evaluated_days.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))
    return paths


def analyze(results_csv: Path, daily_csv: Path | None = None) -> tuple[dict, dict, pd.DataFrame]:
    df = add_test_days(pd.read_csv(results_csv))
    daily = pd.read_csv(daily_csv) if daily_csv and Path(daily_csv).exists() else None

    full = df[~df.truncated]
    anova = {
        "source": str(results_csv),
        "alpha": ALPHA,
        "validation1_reward_oneway": validation1(df, "cumulative_return"),
        "validation1_reward_oneway_sharpe": validation1(df, "sharpe"),
        "validation1_reward_blocked_by_window": validation1_blocked(df, "cumulative_return"),
        "validation1_excluding_truncated": validation1(full, "cumulative_return"),
        "validation2_strategy_oneway": validation2(df),
        "validation2_excluding_truncated_windows": validation2(
            df[~df.window.isin(df.loc[df.truncated, "window"].unique())]
        ),
        "validation3_strategy_x_regime": validation3(daily),
        "kospi": (
            "present" if (df.strategy == "kospi").any()
            else "SKIP in all windows (download failed in Colab; no KOSPI rows in CSV)"
        ),
    }
    trunc = df[df.truncated]
    summary = {
        "source": str(results_csv),
        "n_rows": int(len(df)),
        "windows": window_summary(df),
        "truncated_rows": trunc[["window", "strategy", "seed", "n_days", "ew_days", "cumulative_return", "mdd"]]
        .to_dict(orient="records"),
        "truncation_note": (
            "Test rollouts used PortfolioEnv with Safe-Guard; an MDD>15% terminated the "
            "episode and the shorter series was scored as the test year. Baselines were "
            "scored on the full year. Fixed by rl.backtest.extend_after_safeguard (re-run needed)."
        ),
        "targets": target_check(df),
    }
    return _clean(anova), _clean(summary), df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="docs/results/walk_forward_results_16assets_2015.csv")
    ap.add_argument("--daily", default=None)
    ap.add_argument("--out", default="docs/results")
    args = ap.parse_args()
    out = Path(args.out)
    anova, summary, df = analyze(Path(args.results), args.daily)
    summary["figures"] = make_figures(df, out / "figures")
    (out / "anova_results.json").write_text(json.dumps(anova, indent=2, ensure_ascii=False))
    (out / "wf_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    v1, v2 = anova["validation1_reward_oneway"], anova["validation2_strategy_oneway"]
    print(f"[wf_analysis] V1 reward one-way: F={v1['f_stat']:.3f} p={v1['p_value']:.4f} eta2={v1['eta_squared']:.4f}")
    print(f"[wf_analysis] V2 strategy one-way: F={v2['f_stat']:.3f} p={v2['p_value']:.4f} eta2={v2['eta_squared']:.4f}")
    print(f"[wf_analysis] truncated DRL rows: {len(summary['truncated_rows'])}")
    print(f"[wf_analysis] wrote {out/'anova_results.json'}, {out/'wf_summary.json'}")


if __name__ == "__main__":
    main()
