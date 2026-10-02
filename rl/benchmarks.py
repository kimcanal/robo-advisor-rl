"""시장 벤치마크(S&P500 / KOSPI) 로더 — Walk-Forward·백테스트 12지표 비교용.

우선순위:
  1) 로컬 CSV `rl/data/raw/{SPY|GSPC|KS11}.csv` (Date + Close/Adj Close)
  2) yfinance 다운로드 (SPY 또는 ^GSPC, KOSPI는 ^KS11)

다운로드 실패 시 조용히 건너뛰지 않고 **명확한 메시지와 함께 예외**를 낸다.
로컬 CSV 폴백 경로를 예외 메시지에 포함한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent / "data" / "raw"

# 논리 이름 → (로컬 파일 stem 후보, yfinance 티커 후보)
BENCHMARK_SPECS: dict[str, dict] = {
    "spy": {
        "label": "S&P500 (SPY)",
        "local_stems": ("SPY", "GSPC", "^GSPC"),
        "yf_tickers": ("SPY", "^GSPC"),
    },
    "kospi": {
        "label": "KOSPI (^KS11)",
        "local_stems": ("KS11", "^KS11", "KOSPI"),
        "yf_tickers": ("^KS11",),
    },
}


class BenchmarkDownloadError(RuntimeError):
    """벤치마크 가격을 로컬/원격 모두에서 가져오지 못했을 때."""


def _close_from_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        raise ValueError(f"{path}: 'Close' 또는 'Adj Close' 컬럼이 필요합니다.")
    return df[price_col].astype(float).rename(path.stem)


def _from_local(stems: tuple[str, ...], data_dir: Path) -> pd.Series | None:
    for stem in stems:
        # stem may include ^ — strip for filename friendliness, also try raw
        candidates = [data_dir / f"{stem}.csv", data_dir / f"{stem.lstrip('^')}.csv"]
        for path in candidates:
            if path.exists():
                return _close_from_csv(path)
    return None


def _from_yfinance(tickers: tuple[str, ...], start: str, end: str) -> pd.Series:
    import yfinance as yf

    last_err: Exception | None = None
    for ticker in tickers:
        try:
            raw = yf.download(
                ticker, start=start, end=end, auto_adjust=True, progress=False
            )
            if raw is None or raw.empty:
                last_err = RuntimeError(f"{ticker}: empty download")
                continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if close.empty:
                last_err = RuntimeError(f"{ticker}: Close series empty after dropna")
                continue
            return close.astype(float).rename(ticker)
        except Exception as e:  # noqa: BLE001 — surface in BenchmarkDownloadError
            last_err = e
            continue
    raise BenchmarkDownloadError(
        f"yfinance 벤치마크 다운로드 실패 (tried={tickers}): {last_err}"
    )


def load_benchmark_prices(
    name: str,
    start: str,
    end: str,
    data_dir: Path | str = RAW_DIR,
    *,
    allow_download: bool = True,
) -> pd.Series:
    """벤치마크 수정종가 Series를 반환한다.

    Parameters
    ----------
    name : 'spy' | 'kospi' (대소문자 무시)
    allow_download : False면 로컬 CSV만 사용 (테스트용)
    """
    key = name.lower().strip()
    if key not in BENCHMARK_SPECS:
        raise KeyError(f"지원 벤치마크: {list(BENCHMARK_SPECS)}, got={name!r}")
    spec = BENCHMARK_SPECS[key]
    data_dir = Path(data_dir)

    local = _from_local(spec["local_stems"], data_dir)
    if local is not None:
        return local.loc[start:end].dropna()

    if not allow_download:
        raise BenchmarkDownloadError(
            f"[{spec['label']}] 로컬 CSV 없음 (data_dir={data_dir}, "
            f"stems={spec['local_stems']}). allow_download=False."
        )

    try:
        remote = _from_yfinance(spec["yf_tickers"], start=start, end=end)
        return remote.loc[start:end].dropna()
    except BenchmarkDownloadError:
        raise
    except Exception as e:
        raise BenchmarkDownloadError(
            f"[{spec['label']}] 다운로드 실패: {e}. "
            f"폴백: `{data_dir}/{{SPY|KS11}}.csv`에 Date, Close 컬럼 CSV를 두세요."
        ) from e


def prices_to_log_returns(prices: pd.Series) -> pd.Series:
    """수정종가 → 로그수익률 (백테스트 compute_metrics 입력과 동일 정의)."""
    return np.log(prices / prices.shift(1)).dropna().rename(prices.name)


def load_benchmark_returns(
    name: str,
    start: str,
    end: str,
    data_dir: Path | str = RAW_DIR,
    *,
    allow_download: bool = True,
) -> pd.Series:
    prices = load_benchmark_prices(
        name, start=start, end=end, data_dir=data_dir, allow_download=allow_download
    )
    return prices_to_log_returns(prices)


def load_market_benchmarks(
    start: str,
    end: str,
    data_dir: Path | str = RAW_DIR,
    *,
    allow_download: bool = True,
    names: tuple[str, ...] = ("spy", "kospi"),
) -> dict[str, pd.Series]:
    """S&P500(SPY) + KOSPI 로그수익률 dict. 하나라도 실패하면 예외."""
    out: dict[str, pd.Series] = {}
    errors: list[str] = []
    for name in names:
        try:
            out[name] = load_benchmark_returns(
                name, start=start, end=end, data_dir=data_dir, allow_download=allow_download
            )
        except BenchmarkDownloadError as e:
            errors.append(str(e))
    if errors:
        raise BenchmarkDownloadError(
            "시장 벤치마크 로드 실패:\n- " + "\n- ".join(errors)
        )
    return out


def align_benchmark_returns(
    benchmark: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """백테스트 날짜에 맞춰 벤치마크 수익률을 reindex (결측은 0이 아니라 drop 전에 ffill 금지).

    공통 날짜만 남기려면 호출측에서 intersection을 쓴다. 여기서는 reindex 후
    NaN을 유지해 compute_metrics가 dropna/intersection 하도록 둔다.
    """
    return benchmark.reindex(pd.DatetimeIndex(index))


def synthetic_benchmark_returns(
    index: pd.DatetimeIndex,
    *,
    mu: float = 0.0003,
    sigma: float = 0.01,
    seed: int = 0,
    name: str = "synth",
) -> pd.Series:
    """단위테스트용 합성 로그수익률 (다운로드 불필요)."""
    rng = np.random.default_rng(seed)
    vals = rng.normal(mu, sigma, size=len(index))
    return pd.Series(vals, index=pd.DatetimeIndex(index), name=name, dtype=np.float64)
