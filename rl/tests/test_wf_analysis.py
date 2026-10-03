"""Offline analysis of the committed walk-forward CSV (numbers come only from the CSV)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl.analysis.wf_analysis import add_test_days, analyze, implied_test_days
from rl.curves import convergence_stats, load_monitors
from rl.shap_main import feature_groups, grouped_kernel_shap
from rl.stats_tests import one_way_anova

CSV = Path(__file__).resolve().parents[2] / "docs" / "results" / "walk_forward_results_16assets_2015.csv"


def test_implied_days_inverts_compute_metrics_cagr():
    for n in (35, 98, 252):
        cum = 0.12
        cagr = (1 + cum) ** (252 / n) - 1
        assert implied_test_days(cum, cagr) == pytest.approx(n)


@pytest.mark.skipif(not CSV.exists(), reason="results CSV not present")
def test_committed_results_truncation_and_anova():
    anova, summary, df = analyze(CSV)
    assert len(df) == 84
    assert len(summary["truncated_rows"]) == 19
    assert {r["window"] for r in summary["truncated_rows"]} == {2, 4, 5}
    v2 = anova["validation2_strategy_oneway"]
    assert v2["f_stat"] == pytest.approx(2.584, abs=1e-3)
    assert v2["p_value"] == pytest.approx(0.0911, abs=1e-4)
    assert "tukey_records" not in v2  # p >= 0.05 → no post-hoc
    v1 = anova["validation1_reward_oneway"]
    assert v1["p_value"] > 0.05 and v1["n_per_group"] == {
        "drl_simple": 21, "drl_sharpe": 21, "drl_mdd_penalty": 21}
    assert summary["targets"]["meets_both"] == 0


def test_one_way_anova_runs_tukey_only_when_significant():
    rng = np.random.default_rng(0)
    sig = one_way_anova({"a": pd.Series(rng.normal(0, 1, 30)), "b": pd.Series(rng.normal(2, 1, 30)),
                         "c": pd.Series(rng.normal(0, 1, 30))})
    assert sig["p_value"] < 0.05 and len(sig["tukey_records"]) == 3
    assert {"group1", "group2", "p_adj", "reject"} <= set(sig["tukey_records"][0])
    ns = one_way_anova({"a": pd.Series(rng.normal(0, 1, 30)), "b": pd.Series(rng.normal(0, 1, 30))})
    if ns["p_value"] >= 0.05:
        assert "tukey_records" not in ns


def test_feature_groups_partition_observation():
    tickers, window = ["A", "B", "C"], 5
    names, groups = feature_groups(tickers, window)
    dim = window * 3 + 3 * 3 + 1
    flat = np.sort(np.concatenate(groups))
    np.testing.assert_array_equal(flat, np.arange(dim))
    assert len(names) == 4 * 3 + 1 and names[-1] == "portfolio_risk"


def test_grouped_shap_is_additive():
    tickers, window = ["A", "B"], 3
    names, groups = feature_groups(tickers, window)
    dim = sum(len(g) for g in groups)
    rng = np.random.default_rng(1)
    W = rng.normal(size=dim)
    f = lambda X: np.atleast_2d(X) @ W  # noqa: E731
    bg = rng.normal(size=(15, dim))
    x = rng.normal(size=(1, dim))
    ev, sv = grouped_kernel_shap(f, bg, x, names, groups, nsamples=500)
    assert sv.shape == (1, len(names))
    assert sv.sum() + ev == pytest.approx(f(x)[0], rel=1e-6)
    exact = [(W[g] * (x[0, g] - bg[:, g].mean(0))).sum() for g in groups]
    np.testing.assert_allclose(sv[0], exact, atol=1e-6)


def test_learning_curve_monitor_parsing(tmp_path: Path):
    for rt in ("simple", "mdd_penalty"):
        f = tmp_path / f"w7_{rt}_s0.monitor.csv"
        rows = pd.DataFrame({"r": np.linspace(-1, 1, 50), "l": np.r_[np.full(25, 100), np.full(25, 900)],
                             "t": np.arange(50)})
        f.write_text('#{"t_start": 0}\n' + rows.to_csv(index=False))
    df = load_monitors(tmp_path)
    assert set(df.reward_type) == {"simple", "mdd_penalty"} and df.window.unique().tolist() == [7]
    stats = convergence_stats(df)
    s = next(x for x in stats if x["reward_type"] == "simple")
    assert s["episodes"] == 50 and s["last20_mean_len"] == 900 and s["first20_mean_len"] == 100
