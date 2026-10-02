"""RAG팀 리스크 태그 → RL 관측 공간 인터페이스 (Notion / README 계약).

계약 스키마 (이벤트 행):
    {ticker: str, risk_score: float in [0, 1], tag: str, ts: datetime-like}

리서치·RAG팀이 위험 이벤트를 이 형태로 넘기면, `PortfolioEnv` 관측에
포트폴리오 가중 평균 risk_score 축 하나로 반영한다. 태그가 없어도
기본값 0.0으로 동작해 기존 호출부는 깨지지 않는다.

선택적으로 Safe-Guard MDD 한도를 risk에 비례해 조일 수 있다
(`risk_mdd_scale` > 0, PortfolioEnv).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCHEMA_COLUMNS = ("ticker", "risk_score", "tag", "ts")
DEFAULT_MOCK_PATH = Path(__file__).parent / "data" / "raw" / "risk_tags_mock.csv"


class RiskTagSchemaError(ValueError):
    """리스크 태그 CSV/프레임이 계약 스키마를 만족하지 않을 때."""


def validate_risk_tags(df: pd.DataFrame) -> pd.DataFrame:
    """스키마 검증 후 정규화된 복사본을 반환한다."""
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise RiskTagSchemaError(
            f"리스크 태그 스키마 불일치: 필요 컬럼 {list(SCHEMA_COLUMNS)}, 누락={missing}"
        )
    out = df.loc[:, list(SCHEMA_COLUMNS)].copy()
    out["ticker"] = out["ticker"].astype(str)
    out["tag"] = out["tag"].astype(str)
    out["ts"] = pd.to_datetime(out["ts"], utc=False)
    out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce")
    if out["risk_score"].isna().any():
        raise RiskTagSchemaError("risk_score는 숫자여야 합니다 (NaN 불가).")
    if ((out["risk_score"] < 0) | (out["risk_score"] > 1)).any():
        raise RiskTagSchemaError("risk_score는 [0, 1] 범위여야 합니다.")
    return out.sort_values("ts").reset_index(drop=True)


def load_risk_tags_csv(path: Path | str) -> pd.DataFrame:
    """CSV에서 리스크 태그를 로드한다 (컬럼: ticker, risk_score, tag, ts)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"리스크 태그 CSV 없음: {path}")
    return validate_risk_tags(pd.read_csv(path))


def mock_risk_tags(
    tickers: Iterable[str],
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    n_events_per_ticker: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """단위테스트·데모용 합성 리스크 태그 이벤트."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    if len(dates) == 0:
        raise ValueError("mock_risk_tags: 빈 날짜 범위")
    rows = []
    tags = ["earnings", "geopolitics", "credit", "liquidity", "regulatory"]
    for ticker in tickers:
        n = min(n_events_per_ticker, len(dates))
        picks = rng.choice(dates, size=n, replace=False)
        for ts in sorted(picks):
            rows.append(
                {
                    "ticker": ticker,
                    "risk_score": float(rng.uniform(0.1, 0.95)),
                    "tag": str(rng.choice(tags)),
                    "ts": pd.Timestamp(ts),
                }
            )
    return validate_risk_tags(pd.DataFrame(rows))


def save_risk_tags_csv(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_risk_tags(df).to_csv(path, index=False)
    return path


def risk_score_panel(
    tags: pd.DataFrame | None,
    tickers: list[str],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """이벤트 태그를 (date x ticker) risk_score 패널로 펼친다.

    각 날짜·티커에 대해 **그 날짜 이전(포함) 최신 이벤트**의 risk_score를
    사용한다 (look-ahead 방지). 이벤트가 없으면 0.0.
    """
    panel = pd.DataFrame(
        0.0, index=pd.DatetimeIndex(dates), columns=list(tickers), dtype=np.float64
    )
    if tags is None or len(tags) == 0:
        return panel.astype(np.float32)

    tags = validate_risk_tags(tags)
    left = pd.DataFrame({"date": pd.DatetimeIndex(dates)}).sort_values("date")

    for ticker in tickers:
        sub = tags.loc[tags["ticker"] == ticker].sort_values("ts")
        if sub.empty:
            continue
        right = (
            sub.rename(columns={"ts": "date"})[["date", "risk_score"]]
            .sort_values("date")
            .drop_duplicates("date", keep="last")
        )
        merged = pd.merge_asof(left, right, on="date", direction="backward")
        panel[ticker] = np.nan_to_num(
            merged["risk_score"].to_numpy(dtype=np.float64), nan=0.0
        )

    return panel.astype(np.float32)


def load_or_mock_risk_tags(
    tickers: list[str],
    path: Path | str | None = None,
    *,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    allow_mock: bool = True,
) -> pd.DataFrame:
    """로컬 CSV가 있으면 로드, 없으면(allow_mock) 합성 태그 생성."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    candidates.append(DEFAULT_MOCK_PATH)

    for p in candidates:
        if p.exists():
            return load_risk_tags_csv(p)
    if not allow_mock:
        raise FileNotFoundError(
            f"리스크 태그 CSV를 찾을 수 없습니다. 시도한 경로: {candidates}"
        )
    return mock_risk_tags(tickers, start=start, end=end)
