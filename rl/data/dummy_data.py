"""더미 가격 데이터 생성기.

데이터팀(조윤상)의 실제 수집 파이프라인이 준비되기 전까지, RL 환경/학습 코드를
독립적으로 개발·검증하기 위한 대체 데이터 소스.

실제 데이터로 전환할 때 코드 변경이 필요 없도록, loader.load_price_data()가
반환하는 것과 동일한 형태(DataFrame, index=거래일, columns=티커, 값=수정종가)로
맞춰져 있다. 데이터팀은 이 포맷만 지키면 됨 (자세한 계약은 loader.py 참고).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_dummy_prices(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
    annual_drift: float = 0.06,
    annual_vol: float = 0.20,
    avg_correlation: float = 0.35,
) -> pd.DataFrame:
    """상관관계를 가진 여러 자산의 더미 종가 시계열을 생성한다.

    기하 브라운 운동(GBM) 기반으로, 실제 주가처럼 추세+변동성을 가지되
    자산 간에는 공통 요인(시장 팩터)을 섞어 현실적인 상관구조를 흉내낸다.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    n_assets = len(tickers)

    daily_drift = annual_drift / 252
    daily_vol = annual_vol / np.sqrt(252)

    # 공통 시장 팩터 + 자산별 고유 노이즈를 섞어 상관관계 부여
    market_factor = rng.normal(0, daily_vol, size=n_days)
    idio_noise = rng.normal(0, daily_vol, size=(n_days, n_assets))

    w_common = np.sqrt(avg_correlation)
    w_idio = np.sqrt(1 - avg_correlation)
    shocks = daily_drift + w_common * market_factor[:, None] + w_idio * idio_noise

    # 자산별로 변동성/드리프트를 살짝 다르게 흔들어 완전히 동일한 자산이 되지 않게 함
    asset_vol_scale = rng.uniform(0.7, 1.4, size=n_assets)
    asset_drift_scale = rng.uniform(0.5, 1.6, size=n_assets)
    shocks = shocks * asset_vol_scale + (daily_drift * (asset_drift_scale - 1))

    log_prices = np.cumsum(shocks, axis=0)
    start_price = rng.uniform(20, 300, size=n_assets)
    prices = start_price * np.exp(log_prices)

    df = pd.DataFrame(prices, index=dates, columns=tickers)
    df.index.name = "Date"
    return df


if __name__ == "__main__":
    demo = generate_dummy_prices(["A0", "A1", "A2"], "2019-01-01", "2019-03-01")
    print(demo.head())
