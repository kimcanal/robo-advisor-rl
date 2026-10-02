"""커스텀 Gymnasium 포트폴리오 환경 (요구사항 4-2, 4-3).

관측 공간 : 과거 window일 로그수익률 + 현재 비중 + RSI + MACD 히스토그램
            + 포트폴리오 가중 평균 risk_score (RAG 리스크 태그; 없으면 0.0)
행동 공간 : 연속값 벡터 -> softmax로 비중(합=1, 공매도 없음) 변환
보상      : rewards.py의 3종 변형 중 선택 (reward_type)
Safe-Guard: 누적 낙폭(MDD)이 mdd_limit을 넘으면 에피소드 조기 종료
            (선택) risk_mdd_scale>0 이면 보유 위험도에 비례해 한도를 조임
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from ..config import MDD_LIMIT, SLIPPAGE, TRANSACTION_FEE, WINDOW_SIZE
from ..rewards import get_reward_fn
from ..risk_tags import risk_score_panel


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    # 관측 끝의 리스크 축 개수 (문서화: holdings 가중 평균 risk_score 1축)
    N_RISK_OBS = 1

    def __init__(
        self,
        returns_df,
        rsi_df,
        macd_df,
        window: int = WINDOW_SIZE,
        reward_type: str = "mdd_penalty",
        reward_kwargs: dict | None = None,
        fee: float = TRANSACTION_FEE,
        slippage: float = SLIPPAGE,
        mdd_limit: float = MDD_LIMIT,
        risk_tags: pd.DataFrame | None = None,
        risk_panel: pd.DataFrame | None = None,
        risk_mdd_scale: float = 0.0,
    ):
        super().__init__()
        assert list(returns_df.columns) == list(rsi_df.columns) == list(macd_df.columns)

        self.tickers = list(returns_df.columns)
        self.n_assets = len(self.tickers)
        self.returns = returns_df.to_numpy(dtype=np.float32)
        self.rsi = (rsi_df.to_numpy(dtype=np.float32) / 100.0)  # 0~1 스케일
        self.macd = macd_df.to_numpy(dtype=np.float32)
        self.window = window
        self.reward_type = reward_type
        self.reward_kwargs = reward_kwargs or {}
        self.reward_fn = get_reward_fn(reward_type)
        self.fee_rate = fee + slippage
        self.base_mdd_limit = mdd_limit
        self.mdd_limit = mdd_limit
        self.risk_mdd_scale = float(risk_mdd_scale)

        self.n_steps = self.returns.shape[0]
        if self.n_steps <= self.window + 1:
            raise ValueError("데이터 길이가 window보다 짧습니다.")

        # risk: 사전 계산된 패널이 있으면 그대로, 없으면 태그→패널, 둘 다 없으면 0
        if risk_panel is not None:
            panel = risk_panel.reindex(index=returns_df.index, columns=self.tickers).fillna(0.0)
        elif risk_tags is not None:
            panel = risk_score_panel(risk_tags, self.tickers, returns_df.index)
        else:
            panel = pd.DataFrame(
                0.0, index=returns_df.index, columns=self.tickers, dtype=np.float32
            )
        self.risk = panel.to_numpy(dtype=np.float32)

        # obs = ret_window(flat) + weights + rsi + macd + [mean_holding_risk]
        obs_dim = self.window * self.n_assets + 3 * self.n_assets + self.N_RISK_OBS
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self._reset_state()

    def _reset_state(self):
        self.t = self.window
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.reward_history: list[float] = []
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.mdd_limit = self.base_mdd_limit

    def _portfolio_risk(self, feat_t: int) -> float:
        """현재 비중으로 가중한 보유 자산 risk_score (관측 리스크 축)."""
        return float(np.dot(self.weights, self.risk[feat_t]))

    def _effective_mdd_limit(self, portfolio_risk: float) -> float:
        """risk_mdd_scale>0 이면 보유 위험이 높을수록 Safe-Guard 한도를 조인다.

        effective = base * (1 - scale * portfolio_risk), 하한 base*0.5.
        scale=0(기본)이면 base 그대로 — 기존 동작과 동일.
        """
        if self.risk_mdd_scale <= 0:
            return self.base_mdd_limit
        factor = 1.0 - self.risk_mdd_scale * float(portfolio_risk)
        return float(self.base_mdd_limit * max(0.5, factor))

    def _get_obs(self) -> np.ndarray:
        # returns[t-window:t] 는 현재 바(t)를 제외한 과거 수익률.
        # RSI/MACD/risk는 t-1을 써서 returns[t]와 같은 바의 가격 정보가 관측에
        # 새어 들어가지 않도록 1일 래그 (look-ahead 방지).
        ret_window = self.returns[self.t - self.window : self.t].flatten()
        feat_t = self.t - 1
        portfolio_risk = self._portfolio_risk(feat_t)
        obs = np.concatenate(
            [
                ret_window,
                self.weights,
                self.rsi[feat_t],
                self.macd[feat_t],
                np.asarray([portfolio_risk], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        new_weights = self._softmax(np.asarray(action, dtype=np.float32))

        asset_returns = self.returns[self.t]
        gross_return = float(np.dot(self.weights, asset_returns))

        # /2: 매도 측과 매수 측을 각각 |Δw|에 담으면 같은 거래대금이 두 번
        # 잡힌다(예: A 100%→B 100% 전환 시 sum(|Δw|)=2.0이지만 실제 회전율은
        # 1.0). 표준 관행 및 참고 프로젝트(Dynamic_Regime_Portfolio-luca)의
        # cost_model.py::apply_to_rebalance와 동일하게 /2로 보정.
        turnover = float(np.abs(new_weights - self.weights).sum()) / 2.0
        cost = turnover * self.fee_rate
        net_return = gross_return - cost

        # Compounding consistency with backtest.compute_metrics / _drawdown_series:
        # asset returns are log-returns; portfolio step return is their weighted
        # sum (FinRL-style approx) and is treated as an approximate log-return.
        # Use exp compounding here so Safe-Guard MDD matches backtest MDD.
        self.portfolio_value *= float(np.exp(net_return))
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        reward = self.reward_fn(net_return, self.reward_history, **self.reward_kwargs)
        self.reward_history.append(net_return)
        self.weights = new_weights

        # 리스크 모니터링: 새 비중 기준 포트 위험 → (선택) Safe-Guard 한도 조정
        feat_t = min(self.t, self.n_steps - 1)
        portfolio_risk = self._portfolio_risk(feat_t)
        self.mdd_limit = self._effective_mdd_limit(portfolio_risk)

        self.t += 1
        terminated = drawdown > self.mdd_limit
        truncated = self.t >= self.n_steps

        info = {
            "net_return": net_return,
            "gross_return": gross_return,
            "turnover": turnover,
            "cost": cost,
            "drawdown": drawdown,
            "portfolio_value": self.portfolio_value,
            "safe_guard_triggered": terminated,
            "portfolio_risk": portfolio_risk,
            "effective_mdd_limit": self.mdd_limit,
        }

        obs = self._get_obs() if not (terminated or truncated) else np.zeros_like(self.observation_space.low)
        return obs, float(reward), terminated, truncated, info
