"""yfinance로 실제 ETF 가격을 받아 loader.py 계약(Date, Close 컬럼)에 맞는
CSV로 저장한다. 조윤상님의 정식 데이터 파이프라인이 아니라, RL 모듈을
실제 시장 데이터로 빠르게 시험해보기 위한 임시 스크립트.

사용:
    python -m rl.data.fetch_real_data
"""
from __future__ import annotations

from pathlib import Path

import yfinance as yf

RAW_DIR = Path(__file__).parent / "raw"

# 미션 문서(참고자료)에 예시로 나온 해외 ETF 조합 + 자산군 다양화를 위해 3종 추가
TICKERS = [
    "SPY",  # 미국 대형주
    "QQQ",  # 미국 나스닥 기술주
    "IWM",  # 미국 소형주
    "EFA",  # 선진국(미국 제외) 주식
    "EEM",  # 신�흥국 주식
    "AGG",  # 미국 종합채권
    "TLT",  # 미국 장기국채
    "HYG",  # 하이일드 회사채
    "GLD",  # 금
    "VNQ",  # 리츠(부동산)
]


def fetch(tickers=TICKERS, start="2019-01-01", end=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    for ticker in tickers:
        close = data["Close"][ticker].dropna()
        df = close.reset_index()
        df.columns = ["Date", "Close"]
        out_path = RAW_DIR / f"{ticker}.csv"
        df.to_csv(out_path, index=False)
        print(f"[fetch] {ticker}: {len(df)}행, {df['Date'].min().date()}~{df['Date'].max().date()} -> {out_path}")

    return tickers


if __name__ == "__main__":
    fetch()
