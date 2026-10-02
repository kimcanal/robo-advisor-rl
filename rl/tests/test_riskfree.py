"""riskfree 폴백/로컬 CSV 우선순위 테스트."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rl import riskfree


def test_empty_series_falls_back_to_annual_constant(tmp_path, monkeypatch, capsys):
    """로컬 CSV도 없고 다운로드도 빈 시계열이면 ANNUAL_FALLBACK을 쓰고 0.0이 아니다."""
    empty = tmp_path / "BIL.csv"
    # 존재하지 않는 경로로 강제
    monkeypatch.setattr(riskfree, "BIL_CSV_PATH", empty)

    def boom(*_a, **_k):
        raise RuntimeError("offline")

    monkeypatch.setattr(riskfree, "_from_yfinance", boom)
    rate = riskfree.fetch_risk_free_rate("2019-01-01", "2020-01-01")
    assert rate == pytest.approx(riskfree.ANNUAL_FALLBACK)
    assert rate != 0.0
    err = capsys.readouterr().err
    assert "ANNUAL_FALLBACK" in err


def test_local_csv_preferred_over_yfinance(tmp_path, monkeypatch):
    dates = pd.bdate_range("2019-01-01", periods=60)
    # 일 수익률이 대략 0.04/252 이 되도록 가격 구성
    daily = 0.04 / 252
    prices = (1 + daily) ** pd.Series(range(len(dates)), index=dates)
    csv_path = tmp_path / "BIL.csv"
    pd.DataFrame({"Date": dates, "Close": prices.values}).to_csv(csv_path, index=False)

    monkeypatch.setattr(riskfree, "BIL_CSV_PATH", csv_path)

    def should_not_call(*_a, **_k):
        raise AssertionError("yfinance should not be called when local CSV works")

    monkeypatch.setattr(riskfree, "_from_yfinance", should_not_call)
    rate = riskfree.fetch_risk_free_rate("2019-01-01", "2019-12-31")
    assert rate == pytest.approx(0.04, abs=5e-4)


def test_empty_local_csv_triggers_fallback(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "BIL.csv"
    # Close가 한 점뿐이면 pct_change 후 빈 시계열
    pd.DataFrame({"Date": ["2019-01-02"], "Close": [100.0]}).to_csv(csv_path, index=False)
    monkeypatch.setattr(riskfree, "BIL_CSV_PATH", csv_path)
    monkeypatch.setattr(riskfree, "_from_yfinance", lambda *a, **k: None)
    rate = riskfree.fetch_risk_free_rate("2019-01-01", "2019-12-31")
    assert rate == pytest.approx(riskfree.ANNUAL_FALLBACK)
    assert "ANNUAL_FALLBACK" in capsys.readouterr().err
