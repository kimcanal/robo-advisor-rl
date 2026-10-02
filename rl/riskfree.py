"""무위험이자율을 0 고정 대신 실제 단기국채(BIL) 수익률로.

근거: `Dynamic_Regime_Portfolio-luca` 프로젝트의 CONSTITUTION.md §5.4가
"무위험 수익률은 하드코딩된 상수가 아닌 BIL(단기국채) 또는 SOFR 시계열을
동적으로 사용"하라고 명시한 것과 동일한 결론을, 별도로 참고한 MVO
튜토리얼(10년 국채 수익률 사용)도 내리고 있어 근거가 이중으로 탄탄하다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# fetch/다운로드 모두 실패했을 때 쓰는 문서화된 연율 폴백.
# 0.0으로 조용히 돌아가는 것은 샤프/소르티노를 왜곡하므로 금지.
ANNUAL_FALLBACK = 0.04
BIL_CSV_PATH = Path(__file__).parent / "data" / "raw" / "BIL.csv"


def _annualize_from_close(prices: pd.Series) -> float | None:
    daily_returns = prices.pct_change().dropna()
    if len(daily_returns) == 0:
        return None
    mean = daily_returns.mean()
    # yfinance multi-ticker 형태면 Series가 될 수 있음
    if hasattr(mean, "iloc"):
        mean = mean.iloc[0]
    return float(mean * 252)


def _from_local_csv(start: str, end: str, path: Path = BIL_CSV_PATH) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        return None
    prices = df.loc[start:end, price_col].dropna()
    return _annualize_from_close(prices)


def _from_yfinance(start: str, end: str, proxy: str) -> float | None:
    import yfinance as yf

    prices = yf.download(proxy, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    return _annualize_from_close(prices.dropna())


def fetch_risk_free_rate(start: str, end: str, proxy: str = "BIL") -> float:
    """BIL(1-3개월 단기국채 ETF) 일별 수익률을 연율화한 값을 무위험이자율로 반환.

    우선순위: 로컬 CSV(`data/raw/BIL.csv`) → yfinance 다운로드 →
    문서화된 ANNUAL_FALLBACK(기본 0.04). 실패 시 0.0을 조용히 반환하지 않는다.
    """
    local = _from_local_csv(start, end)
    if local is not None:
        return local

    try:
        remote = _from_yfinance(start, end, proxy)
        if remote is not None:
            return remote
        print(
            f"[riskfree] {proxy} 시계열이 비어 있음 → ANNUAL_FALLBACK={ANNUAL_FALLBACK} 사용",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"[riskfree] {proxy} 조회 실패({e}) → ANNUAL_FALLBACK={ANNUAL_FALLBACK} 사용 "
            f"(로컬 CSV도 없음: {BIL_CSV_PATH})",
            file=sys.stderr,
        )
    return ANNUAL_FALLBACK
