"""yfinance로 실제 ETF 가격을 받아 loader.py 계약(Date, Close 컬럼)에 맞는
CSV로 저장한다. 조윤상님의 정식 데이터 파이프라인이 아니라, RL 모듈을
실제 시장 데이터로 빠르게 시험해보기 위한 임시 스크립트.

사용:
    python -m rl.data.fetch_real_data                       # 10종, 2019-01-01~
    python -m rl.data.fetch_real_data --universe 16 --start 2015-01-01   # PR #7 설정

저장물:
  - data/raw/{ticker}.csv          : Date, Close (자산 + BIL)
  - data/raw/SPY_ohlcv.csv         : Date, Close, Volume (regime.py / walk_forward용)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path(__file__).parent / "raw"

# 미션 문서(참고자료)에 예시로 나온 해외 ETF 조합 + 자산군 다양화를 위해 3종 추가
TICKERS = [
    "SPY",  # 미국 대형주
    "QQQ",  # 미국 나스닥 기술주
    "IWM",  # 미국 소형주
    "EFA",  # 선진국(미국 제외) 주식
    "EEM",  # 신흥국 주식
    "AGG",  # 미국 종합채권
    "TLT",  # 미국 장기국채
    "HYG",  # 하이일드 회사채
    "GLD",  # 금
    "VNQ",  # 리츠(부동산)
]

# PR #7(16종목) 실행에 쓴 추가 6종
EXTRA_6 = ["DIA", "VEA", "BND", "LQD", "SLV", "TIP"]

# 무위험이자율 프록시 — 자산 유니버스에는 넣지 않고 별도 CSV로만 저장
RISK_FREE_TICKER = "BIL"


def _save_close_csv(close: pd.Series, ticker: str) -> Path:
    df = close.dropna().reset_index()
    df.columns = ["Date", "Close"]
    out_path = RAW_DIR / f"{ticker}.csv"
    df.to_csv(out_path, index=False)
    print(f"[fetch] {ticker}: {len(df)}행, {df['Date'].min().date()}~{df['Date'].max().date()} -> {out_path}")
    return out_path


def _save_spy_ohlcv(data) -> Path | None:
    """walk_forward / regime.load_spy_regime이 기대하는 SPY_ohlcv.csv를 저장."""
    try:
        close = data["Close"]["SPY"]
        volume = data["Volume"]["SPY"]
    except Exception as e:
        print(f"[fetch] SPY_ohlcv 저장 실패: {e}")
        return None
    df = pd.DataFrame({"Close": close, "Volume": volume}).dropna().reset_index()
    if df.columns[0] != "Date":
        df = df.rename(columns={df.columns[0]: "Date"})
    out_path = RAW_DIR / "SPY_ohlcv.csv"
    df[["Date", "Close", "Volume"]].to_csv(out_path, index=False)
    print(f"[fetch] SPY_ohlcv: {len(df)}행 -> {out_path}")
    return out_path


def fetch(tickers=TICKERS, start="2019-01-01", end=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_tickers = list(dict.fromkeys([*tickers, RISK_FREE_TICKER]))
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    for ticker in tickers:
        _save_close_csv(data["Close"][ticker], ticker)

    # BIL: riskfree.py가 로컬 CSV를 yfinance보다 우선 사용
    _save_close_csv(data["Close"][RISK_FREE_TICKER], RISK_FREE_TICKER)

    if "SPY" in tickers:
        _save_spy_ohlcv(data)

    return tickers


def fetch_kospi(start="2019-01-01", end=None) -> Path | None:
    """KOSPI 지수를 rl/benchmarks의 다중 소스로 받아 KS11.csv로 저장. 실패하면 원인을 출력."""
    from ..benchmarks import KOSPI_ATTEMPTS, BenchmarkDownloadError, _load_kospi_remote

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        s = _load_kospi_remote(start, end)
    except BenchmarkDownloadError as e:
        print(f"[fetch] KOSPI FAILED: {e}")
        return None
    finally:
        for a in KOSPI_ATTEMPTS:
            print(f"[fetch] KOSPI attempt {a}")
    return _save_close_csv(s, "KS11")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["10", "16"], default="10")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--no-kospi", action="store_true")
    a = ap.parse_args()
    fetch(TICKERS + (EXTRA_6 if a.universe == "16" else []), start=a.start, end=a.end)
    if not a.no_kospi:
        fetch_kospi(start=a.start, end=a.end)
