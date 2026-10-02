"""가격 데이터 -> env 입력(수익률/RSI/MACD) 조립 파이프라인.

train.py, backtest.py, run_demo.py가 공통으로 사용하는 전처리 단계를
한 곳에 모아, "데이터 로드 -> 피처 계산 -> 인덱스 정렬"을 매번 반복하지
않도록 한다.
"""
from __future__ import annotations

import pandas as pd

from .data.loader import load_price_data
from .features import build_feature_frame, log_returns


def prepare_env_inputs(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    use_dummy: bool = True,
    verbose: bool = True,
):
    prices = load_price_data(tickers, start=start, end=end, use_dummy=use_dummy, verbose=verbose)
    returns = log_returns(prices)

    feature_map = build_feature_frame(prices)
    rsi_df = pd.concat({t: feature_map[t]["rsi"] for t in tickers}, axis=1)
    macd_df = pd.concat({t: feature_map[t]["macd_hist"] for t in tickers}, axis=1)

    # 인덱스 교집합만으로는 RSI/MACD 워밍업 구간의 NaN이 걸러지지 않으므로
    # 세 프레임을 합쳐 실제 NaN 유무 기준으로 걸러낸다.
    combined = pd.concat([returns, rsi_df, macd_df], axis=1, sort=False)
    common_index = combined.dropna().index.sort_values()

    returns = returns.loc[common_index]
    rsi_df = rsi_df.loc[common_index]
    macd_df = macd_df.loc[common_index]

    # MACD를 여기서 전체 기간(train+test) 평균/표준편차로 정규화하면 테스트
    # 구간의 통계가 학습 시점 피처에 새어 들어가는 look-ahead bias가 된다
    # (참고 프로젝트 코드 감사에서 동일 패턴을 지적받고 우리 코드에서도 발견).
    # VecNormalize가 학습 데이터만으로 정규화 통계를 계산해 분할 이후에
    # 적용하므로, 원본(raw) MACD를 그대로 넘기고 스케일링은 VecNormalize에 맡긴다.

    return prices.loc[common_index], returns, rsi_df, macd_df
