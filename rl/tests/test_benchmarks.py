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
    df.to_csv(tmp_path / "KS11.csv", index=False)

    spy = load_benchmark_returns("spy", "2020-01-01", "2020-02-15", data_dir=tmp_path, allow_download=False)
    assert len(spy) > 5

    both = load_market_benchmarks(
        "2020-01-01", "2020-02-15", data_dir=tmp_path, allow_download=False
    )
    assert set(both) == {"spy", "kospi"}
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


class _FakeImpersonateError(RuntimeError):
    """curl_cffi ImpersonateError stand-in (chrome150 not supported)."""


def _write_spy_csv(tmp_path: Path, periods: int = 40) -> None:
    idx = pd.bdate_range("2020-01-01", periods=periods)
    prices = 100 * np.exp(
        np.cumsum(np.random.default_rng(0).normal(0.0005, 0.01, size=len(idx)))
    )
    pd.DataFrame({"Date": idx, "Close": prices}).to_csv(tmp_path / "SPY.csv", index=False)


def test_kospi_soft_fail_on_impersonate_error(tmp_path: Path, monkeypatch, capsys):
    """Colab: Impersonating chrome150 is not supported → kospi omit, spy kept."""
    _write_spy_csv(tmp_path)
    import rl.benchmarks as bm

    def selective_yf(tickers, start, end):
        raise BenchmarkDownloadError(
            "yfinance 벤치마크 다운로드 실패 (tried=%s): %s"
            % (tickers, _FakeImpersonateError("Impersonating chrome150 is not supported"))
        )

    monkeypatch.setattr(bm, "_from_yfinance", selective_yf)
    out = load_market_benchmarks(
        "2020-01-01", "2020-02-15", data_dir=tmp_path, allow_download=True, required=("spy",),
    )
    assert "spy" in out and "kospi" not in out
    err = capsys.readouterr().err
    assert "kospi" in err and "soft-fail" in err


def test_kospi_soft_fail_on_empty_download(tmp_path: Path, monkeypatch, capsys):
    """^KS11: empty download → soft-fail kospi, do not abort market load."""
    _write_spy_csv(tmp_path)
    import rl.benchmarks as bm

    def empty_dl(tickers, start, end):
        t0 = tickers[0] if tickers else "?"
        raise BenchmarkDownloadError(
            "yfinance 벤치마크 다운로드 실패 (tried=%s): %s: empty download" % (tickers, t0)
        )

    monkeypatch.setattr(bm, "_from_yfinance", empty_dl)
    out = load_market_benchmarks(
        "2020-01-01", "2020-02-15", data_dir=tmp_path, allow_download=True, required=("spy",),
    )
    assert set(out) == {"spy"}
    assert "empty download" in capsys.readouterr().err


def test_spy_required_still_raises_when_missing(tmp_path: Path, monkeypatch):
    """SPY remains required — both missing → BenchmarkDownloadError."""
    import rl.benchmarks as bm

    def always_fail(tickers, start, end):
        raise BenchmarkDownloadError("fail tried=%s" % (tickers,))

    monkeypatch.setattr(bm, "_from_yfinance", always_fail)
    with pytest.raises(BenchmarkDownloadError) as ei:
        load_market_benchmarks(
            "2020-01-01", "2020-02-15", data_dir=tmp_path, allow_download=True, required=("spy",),
        )
    assert "필수" in str(ei.value) or "fail" in str(ei.value)


def test_kospi_multi_ticker_falls_back_to_ewy(monkeypatch):
    """^KS11 / KS11 fail → EWY succeeds."""
    import rl.benchmarks as bm

    idx = pd.bdate_range("2020-01-01", periods=30)
    prices = pd.Series(
        100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.0004, 0.012, size=len(idx)))),
        index=idx,
        name="EWY",
    )

    def fake_yf(tickers, start, end):
        last = None
        for t in tickers:
            if t in ("^KS11", "KS11"):
                last = RuntimeError("%s: empty download" % t)
                continue
            if t == "EWY":
                return prices.loc[start:end]
            last = RuntimeError("%s: unknown" % t)
        raise BenchmarkDownloadError("fail: %s" % last)

    monkeypatch.setattr(bm, "_from_yfinance", fake_yf)
    rets = load_benchmark_returns(
        "kospi", "2020-01-01", "2020-02-10",
        data_dir=Path("/nonexistent-bench-dir-xyz"), allow_download=True,
    )
    assert len(rets) > 5
    assert rets.name == "EWY"


def test_required_all_means_kospi_hard_fails(tmp_path: Path, monkeypatch):
    """If caller marks kospi required, soft-fail is disabled."""
    _write_spy_csv(tmp_path)
    import rl.benchmarks as bm

    def boom(*_a, **_k):
        raise BenchmarkDownloadError("kospi boom")

    monkeypatch.setattr(bm, "_from_yfinance", boom)
    with pytest.raises(BenchmarkDownloadError):
        load_market_benchmarks(
            "2020-01-01", "2020-02-15", data_dir=tmp_path, allow_download=True,
            required=("spy", "kospi"),
        )
