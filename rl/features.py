"""전처리 및 기술적 지표 계산.

요구사항 4-1(로그 수익률, 결측치 Forward Fill, 정규화)과
4-2(RSI/MACD 2종 이상)를 담당하는 순수 함수 모음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def zscore_normalize(df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Z-score 정규화. window를 주면 롤링, 아니면 전체 샘플 기준."""
    if window is None:
        return (df - df.mean()) / df.std().replace(0, 1e-8)
    mean = df.rolling(window).mean()
    std = df.rolling(window).std().replace(0, 1e-8)
    return ((df - mean) / std).dropna()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean().replace(0, 1e-8)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line})


def build_feature_frame(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """자산별 RSI/MACD 히스토그램을 계산해 티커별 DataFrame으로 정리.

    env가 소비하기 쉬운 형태(티커 -> {feature: series})로 반환한다.
    """
    features = {}
    for ticker in prices.columns:
        rsi_s = rsi(prices[ticker])
        macd_df = macd(prices[ticker])
        features[ticker] = pd.DataFrame(
            {"rsi": rsi_s, "macd_hist": macd_df["hist"]}
        )
    return features
