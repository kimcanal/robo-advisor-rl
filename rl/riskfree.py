"""무위험이자율을 0 고정 대신 실제 단기국채(BIL) 수익률로.

근거: `Dynamic_Regime_Portfolio-luca` 프로젝트의 CONSTITUTION.md §5.4가
"무위험 수익률은 하드코딩된 상수가 아닌 BIL(단기국채) 또는 SOFR 시계열을
동적으로 사용"하라고 명시한 것과 동일한 결론을, 별도로 참고한 MVO
튜토리얼(10년 국채 수익률 사용)도 내리고 있어 근거가 이중으로 탄탄하다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fetch_risk_free_rate(start: str, end: str, proxy: str = "BIL") -> float:
    """BIL(1-3개월 단기국채 ETF) 일별 수익률을 연율화한 값을 무위험이자율로 반환.

    다운로드 실패 시(오프라인, 더미 모드 등) 0.0으로 안전하게 폴백한다.
    """
    try:
        import yfinance as yf

        prices = yf.download(proxy, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        daily_returns = prices.pct_change().dropna()
        if len(daily_returns) == 0:
            return 0.0
        return float(daily_returns.mean().iloc[0] * 252) if hasattr(daily_returns.mean(), "iloc") else float(daily_returns.mean() * 252)
    except Exception as e:
        print(f"[riskfree] BIL 조회 실패({e}), risk_free=0.0으로 폴백")
        return 0.0
