"""시장 국면(Bull/Flat/Bear) 규칙 기반 진단기 — ANOVA 검증3용 (요구사항 4-8).

출처: `~/Dynamic_Regime_Portfolio-luca/models/score_model.py`의
`ScoreRegimeDetector`를 참고/이식. 원본은 S&P500 전종목의 상승비율(breadth)
조건까지 포함하지만, 우리는 개별종목 유니버스 데이터가 없어 **학습이
필요 없는 핵심 4개 기술적 조건**(VWAP, 이동평균 5/20/60, 변동성보정 ROC)만
가져왔다. 학습이 필요 없는 규칙 기반이라 파이프라인에 바로 꽂을 수 있고,
"테스트 구간 평균수익률 부호로 국면을 퉁치던" 기존 방식보다 훨씬
근거 있는 일별 국면 라벨을 준다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MarketRegimeDetector:
    MA_SHORT = 5
    MA_MID = 20
    MA_LONG = 60
    VWAP_WINDOW = 20
    ROC_PERIOD = 20
    ROC_THRESHOLD = 0.02
    BULL_THRESHOLD = 3
    BEAR_THRESHOLD = 3
    SMOOTHING_WINDOW = 5
    SMOOTHING_WINDOW_CRISIS = 2
    VOL_PROXY_WINDOW = 21
    VOL_HIGH_THRESHOLD = 30
    ROC_VOL_WINDOW_SHORT = 20
    ROC_VOL_WINDOW_LONG = 60
    ROC_VOL_CLIP_MIN = 0.5
    ROC_VOL_CLIP_MAX = 3.0

    REGIME_NAMES = {0: "bear", 1: "flat", 2: "bull"}

    def detect(self, close: pd.Series, volume: pd.Series | None = None) -> pd.Series:
        """일별 Bull/Flat/Bear 라벨 시리즈를 반환한다 (close와 같은 인덱스)."""
        close = close.astype("float64")
        volume = pd.Series(1.0, index=close.index) if volume is None else volume.astype("float64")

        vwap = (close * volume).rolling(self.VWAP_WINDOW).sum() / volume.rolling(self.VWAP_WINDOW).sum()
        ma5 = close.rolling(self.MA_SHORT).mean()
        ma20 = close.rolling(self.MA_MID).mean()
        ma60 = close.rolling(self.MA_LONG).mean()
        roc20 = close.pct_change(self.ROC_PERIOD)

        returns = close.pct_change()
        vol_short = returns.rolling(self.ROC_VOL_WINDOW_SHORT).std().shift(1)
        vol_long = returns.rolling(self.ROC_VOL_WINDOW_LONG).std().shift(1)
        vol_ratio = (vol_short / vol_long).clip(self.ROC_VOL_CLIP_MIN, self.ROC_VOL_CLIP_MAX).fillna(1.0)
        roc_thresh = self.ROC_THRESHOLD * vol_ratio

        bull = (
            (close > vwap).astype(int) + (ma5 > ma20).astype(int)
            + (ma20 > ma60).astype(int) + (roc20 > roc_thresh).astype(int)
        )
        bear = (
            (close < vwap).astype(int) + (ma5 < ma20).astype(int)
            + (ma20 < ma60).astype(int) + (roc20 < -roc_thresh).astype(int)
        )

        regime_raw = pd.Series(1, index=close.index)  # 기본값: flat
        regime_raw[(bull >= self.BULL_THRESHOLD) & (bear <= 1)] = 2
        regime_raw[(bear >= self.BEAR_THRESHOLD) & (bull <= 1)] = 0

        # 변동성 급등 구간("위기")은 스무딩 윈도우를 좁혀 국면 전환에 더 민감하게 반응
        vol_proxy = (returns.rolling(self.VOL_PROXY_WINDOW).std() * np.sqrt(252) * 100).shift(1)

        smoothed = []
        for i in range(len(regime_raw)):
            is_crisis = not pd.isna(vol_proxy.iloc[i]) and vol_proxy.iloc[i] > self.VOL_HIGH_THRESHOLD
            window = self.SMOOTHING_WINDOW_CRISIS if is_crisis else self.SMOOTHING_WINDOW
            start = max(0, i - window + 1)
            window_data = regime_raw.iloc[start:i + 1]
            mode_val = window_data.mode()
            smoothed.append(int(mode_val.iloc[0]) if len(mode_val) else int(window_data.iloc[-1]))

        result = pd.Series(smoothed, index=close.index, name="regime")
        return result.map(self.REGIME_NAMES)


def load_spy_regime(path="rl/data/raw/SPY_ohlcv.csv") -> pd.Series:
    """SPY_ohlcv.csv(Close, Volume)로부터 일별 국면 라벨 시리즈를 만든다."""
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    detector = MarketRegimeDetector()
    return detector.detect(df["Close"], df["Volume"])
