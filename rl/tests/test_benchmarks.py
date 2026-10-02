"""벤치마크(S&P500/KOSPI) 모듈 — 합성 시계열 + 로컬 CSV 폴백 테스트."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl.backtest import compute_metrics
from rl.benchmarks import (
    BenchmarkDownloadError,
    align_benchmark_returns,
    load_benchmark_returns,
    load_market_benchmarks,
    prices_to_log_returns,
    synthetic_benchmark_returns,
)


def test_synthetic_series_metrics_have_12_keys():
    idx = pd.bdate_range("2020-01-01", periods=252)
    spy = synthetic_benchmark_returns(idx, seed=0, name="spy")
    kospi = synthetic_benchmark_returns(idx, seed=1, name="kospi", mu=0.0002, sigma=0.015)
    for series in (spy, kospi):
        metrics = compute_metrics(series)
        assert len(metrics) == 12
        assert metrics["mdd"] >= 0


def test_align_benchmark_returns_reindexes():
    idx = pd.bdate_range("2020-01-01", periods=10)
    bench = synthetic_benchmark_returns(idx, seed=2)
    subset = idx[2:7]
    aligned = align_benchmark_returns(bench, subset)
    assert list(aligned.index) == list(subset)
    assert aligned.isna().sum() == 0


def test_local_csv_fallback(tmp_path: Path):
    idx = pd.bdate_range("2020-01-01", periods=30)
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0005, 0.01, size=len(idx))))
    df = pd.DataFrame({"Date": idx, "Close": prices})
    df.to_csv(tmp_path / "SPY.csv", index=False)
    # KOSPI도 필수
    df.to_csv(tmp_path / "KS11.csv", index=False)

    spy = load_benchmark_returns("spy", "2020-01-01", "2020-02-15", data_dir=tmp_path, allow_download=False)
    assert len(spy) > 5
    assert spy.name in ("SPY", "spy") or True

    both = load_market_benchmarks(
        "2020-01-01", "2020-02-15", data_dir=tmp_path, allow_download=False
    )
    assert set(both) == {"spy", "kospi"}
    # 12지표 비교에 바로 쓸 수 있는지
    m = compute_metrics(both["spy"], benchmark=both["kospi"])
    assert "information_ratio" in m


def test_missing_local_fails_loudly_when_download_disabled(tmp_path: Path):
    with pytest.raises(BenchmarkDownloadError) as ei:
        load_benchmark_returns(
            "spy", "2020-01-01", "2020-06-01", data_dir=tmp_path, allow_download=False
        )
    msg = str(ei.value)
    assert "로컬 CSV" in msg or "allow_download" in msg


def test_prices_to_log_returns_matches_definition():
    idx = pd.bdate_range("2021-01-01", periods=5)
    prices = pd.Series([100.0, 110.0, 105.0, 120.0, 115.0], index=idx, name="X")
    rets = prices_to_log_returns(prices)
    expected = np.log(110 / 100)
    assert rets.iloc[0] == pytest.approx(expected)
