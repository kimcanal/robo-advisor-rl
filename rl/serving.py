"""학습된 PPO 정책 → 포트폴리오 비중 (API /optimize method="drl").

아티팩트 레이아웃 (rl.walk_forward --save-dir 가 만들고 rl.export_serving 이 고른다):

    {DRL_ARTIFACT_DIR}/                  기본: <repo>/artifacts/drl
        serving.json                     어떤 run을 서빙하는지 + 스냅샷 관측
        models/{run_tag}/model.zip       SB3 PPO
        models/{run_tag}/vecnorm.pkl     VecNormalize (obs_rms만 사용)
        models/{run_tag}/meta.json       tickers, window, 학습/테스트 구간
        models/{run_tag}/test_daily.csv  테스트 구간 일별 수익률·보유 비중
        monitor/*.monitor.csv            학습 곡선
        shap/*.json, shap/*.png          본학습 모델 SHAP
        lambda_sweep.csv                 변형3 lambda 곡선

추론 경로 = 학습 환경과 동일:
    raw obs (PortfolioEnv._get_obs와 같은 배치) → VecNormalize obs 정규화
    → PPO.predict(deterministic) → clip[-1,1] → softmax → 비중
시드 앙상블: 같은 보상·윈도우의 시드별 비중을 산술평균한다.

이 모듈은 stable_baselines3를 함수 안에서만 import 한다 (테스트·API 기동 시 torch 불필요).
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_feature_frame, log_returns

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "drl"
# 서빙 보상은 테스트 성과로 고르지 않는다(선택 편향). 명세 기본(변형3, lambda=1.0).
DEFAULT_SERVING_REWARD = "mdd_penalty"
MIN_HISTORY_EXTRA = 40  # MACD(26+9)·RSI(14) 워밍업 여유


def artifact_dir() -> Path:
    return Path(os.getenv("DRL_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)))


# ---------------------------------------------------------------- observation
def build_observation(
    prices: pd.DataFrame,
    tickers: list[str],
    window: int,
    current_weights: np.ndarray | None = None,
    portfolio_risk: float = 0.0,
) -> tuple[np.ndarray, pd.Timestamp]:
    """가격 패널(마지막 행 = 기준일 T)로 PortfolioEnv와 같은 raw 관측을 만든다.

    PortfolioEnv는 스텝 t에서 returns[t-window:t]와 features[t-1]을 본다.
    데이터가 T까지 있을 때의 "다음 거래일 비중 결정"은 t = T+1에 해당하므로
    returns의 마지막 window행(… T)과 T일 RSI/MACD를 쓴다.
    """
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"prices missing tickers: {missing}")
    prices = prices[tickers].sort_index().ffill().dropna()
    returns = log_returns(prices)
    fmap = build_feature_frame(prices)
    rsi_df = pd.concat({t: fmap[t]["rsi"] for t in tickers}, axis=1)
    macd_df = pd.concat({t: fmap[t]["macd_hist"] for t in tickers}, axis=1)
    common = pd.concat([returns, rsi_df, macd_df], axis=1, sort=False).dropna().index.sort_values()
    if len(common) < window:
        raise ValueError(
            f"need >= {window} rows after feature warm-up, got {len(common)} "
            f"(send >= {window + MIN_HISTORY_EXTRA} price rows)"
        )
    returns, rsi_df, macd_df = returns.loc[common], rsi_df.loc[common], macd_df.loc[common]
    n = len(tickers)
    w = np.full(n, 1.0 / n, dtype=np.float32) if current_weights is None else np.asarray(
        current_weights, dtype=np.float32
    )
    if w.shape != (n,):
        raise ValueError(f"current_weights must have length {n}")
    obs = np.concatenate(
        [
            returns.to_numpy(dtype=np.float32)[-window:].flatten(),
            w,
            rsi_df.to_numpy(dtype=np.float32)[-1] / 100.0,
            macd_df.to_numpy(dtype=np.float32)[-1],
            np.asarray([portfolio_risk], dtype=np.float32),
        ]
    ).astype(np.float32)
    return obs, common[-1]


# --------------------------------------------------------------- normalization
@dataclass
class ObsNorm:
    mean: np.ndarray
    var: np.ndarray
    clip_obs: float = 10.0
    epsilon: float = 1e-8

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(x, -self.clip_obs, self.clip_obs).astype(np.float32)

    @classmethod
    def from_vecnormalize_pickle(cls, path: Path) -> "ObsNorm":
        """VecNormalize.save()가 만든 pickle에서 obs_rms만 꺼낸다 (venv 불필요)."""
        with open(path, "rb") as f:
            vn = pickle.load(f)
        rms = vn.obs_rms
        return cls(
            mean=np.asarray(rms.mean, dtype=np.float64),
            var=np.asarray(rms.var, dtype=np.float64),
            clip_obs=float(getattr(vn, "clip_obs", 10.0)),
            epsilon=float(getattr(vn, "epsilon", 1e-8)),
        )


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def action_to_weights(action: np.ndarray) -> np.ndarray:
    """SB3 predict(Box)는 이미 [-1,1]로 clip — 같은 변환을 한 번 더 해도 무해."""
    return softmax(np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0))


# --------------------------------------------------------------------- policy
@dataclass
class PolicyRun:
    run_dir: Path
    meta: dict
    norm: ObsNorm
    model: object = None  # SB3 PPO (lazy) 또는 테스트용 duck-typed .predict

    @property
    def tickers(self) -> list[str]:
        return list(self.meta["tickers"])

    @property
    def window(self) -> int:
        return int(self.meta["window"])

    def load_model(self):
        if self.model is None:
            from stable_baselines3 import PPO  # heavy import, only on first use

            self.model = PPO.load(str(self.run_dir / "model.zip"), device="cpu")
        return self.model

    def weights(self, raw_obs: np.ndarray) -> np.ndarray:
        model = self.load_model()
        action, _ = model.predict(self.norm(raw_obs)[None, :], deterministic=True)
        return action_to_weights(np.asarray(action)[0])


def load_run(run_dir: Path) -> PolicyRun:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    norm = ObsNorm.from_vecnormalize_pickle(run_dir / "vecnorm.pkl")
    return PolicyRun(run_dir=run_dir, meta=meta, norm=norm)


def discover_runs(root: Path | None = None) -> list[dict]:
    """models/*/meta.json 목록 (model.zip·vecnorm.pkl이 둘 다 있는 것만)."""
    root = root or artifact_dir()
    out = []
    for meta_path in sorted((root / "models").glob("*/meta.json")):
        d = meta_path.parent
        if (d / "model.zip").is_file() and (d / "vecnorm.pkl").is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["run_dir"] = str(d)
            out.append(meta)
    return out


def serving_config(root: Path | None = None) -> dict | None:
    root = root or artifact_dir()
    p = root / "serving.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


@dataclass
class DRLServer:
    """보상 1종 × 시드 N개 앙상블. 로드 실패/아티팩트 없음은 status로 알린다."""

    root: Path = field(default_factory=artifact_dir)
    runs: list[PolicyRun] = field(default_factory=list)
    config: dict | None = None

    @classmethod
    def from_artifacts(cls, root: Path | None = None) -> "DRLServer":
        root = Path(root) if root else artifact_dir()
        cfg = serving_config(root)
        if cfg and cfg.get("run_tags"):
            dirs = [root / "models" / tag for tag in cfg["run_tags"]]
        else:
            metas = discover_runs(root)
            reward = os.getenv("DRL_SERVING_REWARD", DEFAULT_SERVING_REWARD)
            metas = [m for m in metas if m.get("reward_type") == reward]
            if metas:
                last_test = max(m["test_end"] for m in metas)
                metas = [m for m in metas if m["test_end"] == last_test]
            dirs = [Path(m["run_dir"]) for m in metas]
        runs = [load_run(d) for d in dirs if (d / "model.zip").is_file()]
        return cls(root=root, runs=runs, config=cfg)

    @property
    def available(self) -> bool:
        return bool(self.runs)

    def status(self) -> dict:
        if not self.runs:
            return {
                "drl_model_available": False,
                "artifact_dir": str(self.root),
                "reason": "no models/*/model.zip + vecnorm.pkl under artifact_dir",
            }
        m = self.runs[0].meta
        return {
            "drl_model_available": True,
            "artifact_dir": str(self.root),
            "reward_type": m.get("reward_type"),
            "reward_kwargs": m.get("reward_kwargs"),
            "seeds": [r.meta.get("seed") for r in self.runs],
            "train": [m.get("train_start"), m.get("train_end")],
            "test": [m.get("test_start"), m.get("test_end")],
            "timesteps": m.get("timesteps"),
            "tickers": self.runs[0].tickers,
            "models_loaded": [r.model is not None for r in self.runs],
        }

    def snapshot_obs(self) -> tuple[np.ndarray, str] | None:
        snap = (self.config or {}).get("snapshot")
        if not snap:
            return None
        return np.asarray(snap["obs_raw"], dtype=np.float32), str(snap["as_of"])

    def predict(self, prices: pd.DataFrame | None = None,
                current_weights: np.ndarray | None = None) -> dict:
        if not self.runs:
            raise RuntimeError("no trained DRL policy available")
        t0 = time.perf_counter()
        tickers, window = self.runs[0].tickers, self.runs[0].window
        if prices is not None:
            obs, as_of = build_observation(prices, tickers, window, current_weights)
            source = "prices"
            as_of = str(pd.Timestamp(as_of).date())
        else:
            snap = self.snapshot_obs()
            if snap is None:
                raise RuntimeError("no prices given and no snapshot in serving.json")
            obs, as_of = snap
            if current_weights is not None:
                n = len(tickers)
                obs = obs.copy()
                obs[window * n: window * n + n] = np.asarray(current_weights, dtype=np.float32)
            source = "snapshot"
        per_seed = [r.weights(obs) for r in self.runs]
        ens = np.mean(per_seed, axis=0)
        ens = ens / ens.sum()
        return {
            "tickers": tickers,
            "weights": {t: float(w) for t, w in zip(tickers, ens)},
            "per_seed": {
                f"seed{r.meta.get('seed')}": {t: float(w) for t, w in zip(tickers, ps)}
                for r, ps in zip(self.runs, per_seed)
            },
            "as_of": as_of,
            "data_source": source,
            "reward_type": self.runs[0].meta.get("reward_type"),
            "run_tags": [r.meta.get("run_tag") for r in self.runs],
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }


def load_local_prices(tickers: list[str], data_dir: Path | None = None,
                      tail: int = 400) -> pd.DataFrame | None:
    """rl/data/raw/{ticker}.csv가 전부 있을 때만 최근 tail행 가격 패널을 반환."""
    from .data.loader import DEFAULT_DATA_DIR, _load_local_csv

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    series = []
    for t in tickers:
        s = _load_local_csv(t, data_dir)
        if s is None:
            return None
        series.append(s)
    return pd.concat(series, axis=1).sort_index().ffill().dropna().tail(tail)
