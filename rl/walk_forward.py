"""Walk-Forward 백테스트: 학습 4년 -> 테스트 1년, 윈도우 2개 이상, 윈도우
이동 시 모델 재학습 (요구사항 4-5) — 지금까지 없었던 항목이라 새로 추가.

일별 시장 국면(Bull/Flat/Bear) 라벨을 `regime.py`(규칙기반, 학습 불필요)로
매겨서, 검증3(시장 국면별 전략 성과 비교, Two-way ANOVA)을 실제 일별 데이터로
수행한다. 이전 버전은 "테스트 구간 평균수익률 부호"로 윈도우당 라벨 1개를
퉁쳤는데, 이번엔 일별 라벨 x 일별 수익률이라 통계적으로 훨씬 탄탄하다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from .backtest import compute_metrics, walk_forward_windows
from .benchmarks import (
    BenchmarkDownloadError,
    align_benchmark_returns,
    load_market_benchmarks,
    synthetic_benchmark_returns,
)
from .config import WINDOW_SIZE
from .env.portfolio_env import PortfolioEnv
from .mvo import equal_weight_backtest, rolling_mvo_backtest
from .pipeline import prepare_env_inputs
from .regime import MarketRegimeDetector, load_spy_regime
from .riskfree import fetch_risk_free_rate
from .stats_tests import one_way_anova, two_way_anova
from .vecnorm import make_eval_vecnorm, make_train_vecnorm, rollout_vecnorm

OUT_DIR = Path(__file__).parent / "outputs" / "walk_forward"
REAL_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "TLT", "HYG", "GLD", "VNQ"]
SPY_OHLCV_PATH = Path(__file__).parent / "data" / "raw" / "SPY_ohlcv.csv"


def _load_benchmarks_for_run(start: str, end: str, use_dummy: bool) -> dict:
    """S&P500(SPY) + KOSPI 로그수익률.

    로컬 CSV → yfinance 순. spy는 필수; kospi는 soft-fail(생략) 가능.
    필수(spy)까지 실패 시: 실데이터 모드면 예외, 더미 모드면 합성 폴백.
    """
    try:
        # required=("spy",): ^KS11 ImpersonateError/empty여도 WF 전체가 죽지 않음
        return load_market_benchmarks(start, end, allow_download=True, required=("spy",))
    except BenchmarkDownloadError as e:
        if not use_dummy:
            raise
        print(f"[walk_forward] 벤치마크 로드 실패 → 합성 시계열 폴백 ({e})")
        idx = pd.bdate_range(start, end)
        return {
            "spy": synthetic_benchmark_returns(idx, mu=0.00035, sigma=0.01, seed=1, name="spy"),
            "kospi": synthetic_benchmark_returns(idx, mu=0.00025, sigma=0.012, seed=2, name="kospi"),
        }



def _build_daily_regime(returns_df: pd.DataFrame) -> pd.Series:
    """SPY_ohlcv.csv가 있으면 실제 SPY로, 없으면(더미 등) 동일가중 벤치마크
    가격을 합성해서 같은 규칙기반 탐지기로 국면 라벨을 매긴다."""
    if SPY_OHLCV_PATH.exists():
        regime = load_spy_regime(SPY_OHLCV_PATH)
    else:
        bench_price = (1 + returns_df.mean(axis=1)).cumprod()
        regime = MarketRegimeDetector().detect(bench_price)
    return regime.reindex(returns_df.index).ffill().bfill()


def _save_incremental(rows: list, daily_rows: list) -> None:
    """윈도우/시드가 끝날 때마다 CSV를 덮어써서 장시간 실행 중 유실을 막는다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "walk_forward_results.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(OUT_DIR / "walk_forward_daily.csv", index=False)


def run(
    tickers=None, use_dummy=True, start="2019-01-01", end="2024-12-31",
    window=WINDOW_SIZE, timesteps=120_000, seeds=(0,),
    reward_types=("simple", "sharpe", "mdd_penalty"),
    train_years=4, test_years=1, n_windows=2,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = tickers or [f"A{i}" for i in range(10)]
    _, returns, rsi_df, macd_df = prepare_env_inputs(tickers, start=start, end=end, use_dummy=use_dummy, verbose=False)

    risk_free = fetch_risk_free_rate(start, end) if not use_dummy else 0.0
    print(f"[walk_forward] risk_free_rate={risk_free:.4f} (연율화, BIL 기준)")

    market_benchmarks = _load_benchmarks_for_run(start, end, use_dummy=use_dummy)
    print(f"[walk_forward] market benchmarks loaded: {list(market_benchmarks)}")

    windows = walk_forward_windows(returns.index, train_years=train_years, test_years=test_years, n_windows=n_windows)
    if not windows:
        raise ValueError("데이터 범위가 부족해 Walk-Forward 윈도우를 만들 수 없습니다.")

    daily_regime = _build_daily_regime(returns)

    rows = []
    daily_rows = []
    for w_idx, w in enumerate(windows):
        train_returns = returns.loc[w["train_start"]:w["train_end"]]
        train_rsi = rsi_df.loc[w["train_start"]:w["train_end"]]
        train_macd = macd_df.loc[w["train_start"]:w["train_end"]]

        test_start_pos = returns.index.get_indexer([w["test_start"]], method="bfill")[0]
        test_end_pos = returns.index.get_indexer([w["test_end"]], method="ffill")[0]
        buffer_start_pos = max(0, test_start_pos - window)
        test_slice_idx = returns.index[buffer_start_pos: test_end_pos + 1]

        test_returns = returns.loc[test_slice_idx]
        test_rsi = rsi_df.loc[test_slice_idx]
        test_macd = macd_df.loc[test_slice_idx]
        test_dates_no_buffer = test_returns.index[window:]

        regime_counts = daily_regime.loc[test_dates_no_buffer].value_counts()
        print(
            f"[walk_forward] window{w_idx+1}: train {w['train_start'].date()}~{w['train_end'].date()}, "
            f"test {w['test_start'].date()}~{w['test_end'].date()} (국면 분포: {regime_counts.to_dict()})"
        )

        for rt in reward_types:
            reward_kwargs = {"mdd_lambda": 1.0} if rt == "mdd_penalty" else {}
            for seed in seeds:
                # default-arg binding으로 루프 변수 late-binding 방지
                train_env_fn = lambda rt=rt, reward_kwargs=reward_kwargs: PortfolioEnv(
                    train_returns, train_rsi, train_macd, window=window,
                    reward_type=rt, reward_kwargs=reward_kwargs,
                )
                train_venv = make_train_vecnorm(train_env_fn)
                model = PPO("MlpPolicy", train_venv, verbose=0, seed=seed)
                model.learn(total_timesteps=timesteps)

                test_env_fn = lambda rt=rt, reward_kwargs=reward_kwargs: PortfolioEnv(
                    test_returns, test_rsi, test_macd, window=window,
                    reward_type=rt, reward_kwargs=reward_kwargs,
                )
                eval_venv = make_eval_vecnorm(test_env_fn, train_venv)
                rets, _ = rollout_vecnorm(model, eval_venv)
                dates = test_dates_no_buffer[: len(rets)]
                rets_s = pd.Series(rets, index=dates)
                spy_aligned = align_benchmark_returns(market_benchmarks["spy"], dates)
                metrics = compute_metrics(rets_s, benchmark=spy_aligned, risk_free=risk_free)
                rows.append({"window": w_idx + 1, "strategy": f"drl_{rt}", "seed": seed, **metrics})
                print(f"[walk_forward] window{w_idx+1} {rt} seed{seed} -> return={metrics['cumulative_return']:.4f}, mdd={metrics['mdd']:.4f}")

                for d, r in zip(dates, rets):
                    daily_rows.append({"window": w_idx + 1, "strategy": f"drl_{rt}", "seed": seed, "date": d, "daily_return": r, "regime": daily_regime.loc[d]})

                _save_incremental(rows, daily_rows)

        combined_for_mvo = pd.concat([train_returns, test_returns.iloc[window:]])
        mvo_ret = rolling_mvo_backtest(combined_for_mvo).iloc[-len(test_returns.iloc[window:]):]
        equal_ret = equal_weight_backtest(test_returns.iloc[window:])
        test_idx = test_returns.iloc[window:].index
        spy_ret = align_benchmark_returns(market_benchmarks["spy"], test_idx).dropna()
        # kospi may be absent after soft-fail (ImpersonateError / empty ^KS11)
        kospi_series = market_benchmarks.get("kospi")
        kospi_ret = (
            align_benchmark_returns(kospi_series, test_idx).dropna()
            if kospi_series is not None
            else pd.Series(dtype=float)
        )
        if kospi_series is None:
            print(f"[walk_forward] window{w_idx+1} kospi → SKIP (benchmark soft-fail / not loaded)")
        # 12-metric comparison vs DRL: EW/MVO + market benchmarks (S&P500, KOSPI)
        baselines = [
            ("mvo", mvo_ret),
            ("equal_weight", equal_ret),
            ("spy", spy_ret),
            ("kospi", kospi_ret),
        ]
        for name, series in baselines:
            if len(series) == 0:
                print(f"[walk_forward] window{w_idx+1} {name} -> SKIP (no overlapping dates)")
                continue
            bench_for_alpha = spy_ret if name not in ("spy",) else None
            metrics = compute_metrics(series, benchmark=bench_for_alpha, risk_free=risk_free)
            rows.append({"window": w_idx + 1, "strategy": name, "seed": None, **metrics})
            print(f"[walk_forward] window{w_idx+1} {name} -> return={metrics['cumulative_return']:.4f}, mdd={metrics['mdd']:.4f}")
            for d, r in series.items():
                if d in daily_regime.index:
                    daily_rows.append({"window": w_idx + 1, "strategy": name, "seed": None, "date": d, "daily_return": r, "regime": daily_regime.loc[d]})

        _save_incremental(rows, daily_rows)

    df = pd.DataFrame(rows)
    daily_df = pd.DataFrame(daily_rows)
    print("\n[walk_forward] 전체 결과 (윈도우별 요약):")
    print(df[["window", "strategy", "seed", "cumulative_return", "sharpe", "mdd"]])

    groups = {
        name: df.loc[df.strategy == name, "cumulative_return"]
        for name in ["drl_mdd_penalty", "mvo", "equal_weight"]
    }
    anova2 = one_way_anova(groups)
    print(
        f"\n[walk_forward] 검증2 ANOVA(DRL vs MVO vs 동일가중, 윈도우 풀링): "
        f"F={anova2['f_stat']:.3f}, p={anova2['p_value']:.4f}, eta^2={anova2['eta_squared']:.4f}"
    )

    regime_var = daily_df["regime"].nunique()
    if regime_var >= 2 and daily_df["strategy"].nunique() >= 2:
        anova3 = two_way_anova(daily_df, value_col="daily_return", factor1="strategy", factor2="regime")
        print("\n[walk_forward] 검증3 Two-way ANOVA(전략 x 시장국면, 일별 수익률):")
        print(anova3["anova_table"])
        print(f"eta^2: {anova3['eta_squared']}")
    else:
        print(f"\n[walk_forward] 검증3 생략: 국면 종류가 {regime_var}개뿐이라 Two-way ANOVA 불가 (더 긴 기간 필요)")

    print(f"[walk_forward] saved -> {OUT_DIR / 'walk_forward_results.csv'}, {OUT_DIR / 'walk_forward_daily.csv'}")
    return df, daily_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    if args.real:
        run(tickers=REAL_TICKERS, use_dummy=False, start="2019-01-01", end="2026-09-29", timesteps=args.timesteps, seeds=seeds)
    else:
        run(timesteps=args.timesteps, seeds=seeds)
