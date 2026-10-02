"""커스텀 Gymnasium 포트폴리오 환경 (요구사항 4-2, 4-3).

관측 공간 : 과거 window일 로그수익률 + 현재 비중 + RSI + MACD 히스토그램
행동 공간 : 연속값 벡터 -> softmax로 비중(합=1, 공매도 없음) 변환
보상      : rewards.py의 3종 변형 중 선택 (reward_type)
Safe-Guard: 누적 낙폭(MDD)이 mdd_limit을 넘으면 에피소드 조기 종료
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..config import MDD_LIMIT, SLIPPAGE, TRANSACTION_FEE, WINDOW_SIZE
from ..rewards import get_reward_fn


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

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
        self.mdd_limit = mdd_limit

        self.n_steps = self.returns.shape[0]
        if self.n_steps <= self.window + 1:
            raise ValueError("데이터 길이가 window보다 짧습니다.")

        obs_dim = self.window * self.n_assets + 3 * self.n_assets
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

    def _get_obs(self) -> np.ndarray:
        # returns[t-window:t] 는 현재 바(t)를 제외한 과거 수익률.
        # RSI/MACD는 t-1을 써서 returns[t]와 같은 바의 가격 정보가 관측에
        # 새어 들어가지 않도록 1일 래그 (look-ahead 방지).
        ret_window = self.returns[self.t - self.window : self.t].flatten()
        feat_t = self.t - 1
        obs = np.concatenate(
            [ret_window, self.weights, self.rsi[feat_t], self.macd[feat_t]]
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
        }

        obs = self._get_obs() if not (terminated or truncated) else np.zeros_like(self.observation_space.low)
        return obs, float(reward), terminated, truncated, info
