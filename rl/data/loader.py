"""가격 데이터 로더 — 데이터팀(조윤상) 파이프라인과의 인터페이스 계약 지점.

계약(Contract):
    data/raw/{ticker}.csv 에 다음 컬럼을 포함한 CSV를 두면 자동으로 인식한다.
        - Date (또는 index로 파싱 가능한 날짜 컬럼)
        - Close 또는 Adj Close (수정종가)
    이 형식만 지키면 이 파일의 나머지 코드나 env/train/backtest 쪽은
    전혀 수정할 필요가 없다. 실제 데이터가 없는 티커는 자동으로 더미 데이터로
    대체되므로, 데이터팀 작업이 끝나는 대로 파일만 채워 넣으면 됨.

우선순위: 로컬 CSV(data/raw) > yfinance 실시간 수집(선택) > 더미 데이터.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dummy_data import generate_dummy_prices

DEFAULT_DATA_DIR = Path(__file__).parent / "raw"


def _load_local_csv(ticker: str, data_dir: Path) -> pd.Series | None:
    path = data_dir / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        raise ValueError(f"{path}: 'Close' 또는 'Adj Close' 컬럼이 필요합니다.")
    return df[price_col].rename(ticker)


def load_price_data(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    data_dir: Path | str = DEFAULT_DATA_DIR,
    use_dummy: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """티커 리스트에 대한 수정종가 DataFrame을 반환한다.

    각 티커별로 로컬 CSV가 있으면 그것을 쓰고, 없으면(그리고 use_dummy=True면)
    더미 데이터로 자동 대체한다. 실제/더미가 섞여도 동작하지만, 어떤 티커가
    더미로 대체됐는지는 verbose=True일 때 콘솔에 경고로 표시한다.
    """
    data_dir = Path(data_dir)
    series_list = []
    dummy_tickers = []

    for ticker in tickers:
        s = _load_local_csv(ticker, data_dir)
        if s is None:
            dummy_tickers.append(ticker)
        else:
            series_list.append(s)

    if dummy_tickers:
        if not use_dummy:
            raise FileNotFoundError(
                f"실제 데이터가 없는 티커: {dummy_tickers} (use_dummy=False)"
            )
        if verbose:
            print(
                f"[loader] 실제 CSV 없음 → 더미 데이터로 대체: {dummy_tickers} "
                f"(data_dir={data_dir})"
            )
        dummy_df = generate_dummy_prices(dummy_tickers, start=start, end=end)
        series_list.append(dummy_df)

    prices = pd.concat(
        [s if isinstance(s, pd.DataFrame) else s.to_frame() for s in series_list],
        axis=1,
    )
    prices = prices.loc[start:end].ffill().dropna()
    return prices[tickers]
