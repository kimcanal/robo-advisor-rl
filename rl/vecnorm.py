"""VecNormalize 래퍼 — 관측값/보상 정규화 (PPO 학습 효율 개선).

우리 일별 수익률은 매우 작은 스케일(~0.001~0.02)이라 PPO가 암묵적으로
기대하는 대략 단위 스케일 보상과 맞지 않아 학습 신호가 약할 수 있다.
VecNormalize로 관측값·보상을 평균0/분산1로 정규화하는 건 연속제어 RL에서
거의 표준 관행(OpenAI Baselines, SB3 공식 예제 다수가 기본 적용)인데,
지금까지 우리 파이프라인엔 빠져 있었다.

평가(rollout) 시 규칙:
  - norm_reward=False : 백테스트 지표는 정규화된 보상이 아니라 실제
    수익률(PortfolioEnv의 info['net_return'])을 써야 하므로 무관하지만,
    혹시 reward 자체를 쓰는 코드가 있을 경우를 대비해 평가 중엔 꺼둔다.
  - training=False     : 테스트 구간 데이터로 정규화 통계(평균/분산)를
    갱신하면 미래 정보 누수이므로, 학습 때 쌓인 obs_rms를 고정해서 재사용.
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_train_vecnorm(env_fn, gamma: float = 0.99) -> VecNormalize:
    venv = DummyVecEnv([env_fn])
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=gamma)


def make_eval_vecnorm(env_fn, train_vecnorm: VecNormalize) -> VecNormalize:
    """train_vecnorm의 관측값 정규화 통계(obs_rms)를 고정해서 그대로 쓰는
    평가용 VecNormalize를 만든다. 통계가 다르면 학습된 정책이 보는 입력
    분포와 평가 시 입력 분포가 달라져 성능이 깨진다."""
    venv = DummyVecEnv([env_fn])
    eval_venv = VecNormalize(
        venv, norm_obs=True, norm_reward=False, clip_obs=10.0,
        training=False, gamma=train_vecnorm.gamma,
    )
    eval_venv.obs_rms = train_vecnorm.obs_rms
    return eval_venv


def rollout_vecnorm(model, eval_venv: VecNormalize):
    """VecNormalize로 감싼 평가 환경에서 1 에피소드를 deterministic하게
    굴리고, (실제 net_return 배열, 정규화된 관측값 배열)을 반환한다."""
    obs = eval_venv.reset()
    returns = []
    obs_history = []
    done = False
    while not done:
        obs_history.append(obs[0].copy())
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, dones, infos = eval_venv.step(action)
        done = bool(dones[0])
        returns.append(infos[0]["net_return"])
    return np.array(returns), np.array(obs_history)
