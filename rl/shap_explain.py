"""SHAP 기반 의사결정 해석 (요구사항 4-7).

RL 정책은 다차원 벡터(비중)를 출력하므로, SHAP은 "특정 자산 하나의 비중
결정"을 스칼라 타깃으로 잡아서 설명한다 (기본값: 비중이 가장 큰 자산).

주의: KernelExplainer는 입력 차원(윈도우 x 자산수)이 커지면 느려진다.
데모 단계에서는 자산 수/윈도우를 작게 유지하고, 스케일업 시에는
nsamples를 줄이거나 shap.sample로 배경 데이터를 서브샘플링할 것.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap


def build_feature_names(tickers: list[str], window: int, *, include_risk: bool = True) -> list[str]:
    names = []
    for k in range(window, 0, -1):
        for t in tickers:
            names.append(f"ret_t-{k}_{t}")
    names += [f"weight_{t}" for t in tickers]
    names += [f"rsi_{t}" for t in tickers]
    names += [f"macd_{t}" for t in tickers]
    # PortfolioEnv 관측 끝의 리스크 축 (보유 비중 가중 평균 risk_score)
    if include_risk:
        names.append("portfolio_risk")
    return names


def _make_scalar_model_fn(sb3_model, asset_index: int):
    def _predict(obs_batch: np.ndarray) -> np.ndarray:
        actions, _ = sb3_model.predict(obs_batch, deterministic=True)
        actions = np.atleast_2d(actions)
        exp = np.exp(actions - actions.max(axis=1, keepdims=True))
        weights = exp / exp.sum(axis=1, keepdims=True)
        return weights[:, asset_index]

    return _predict


def explain_decision(
    sb3_model,
    background_obs: np.ndarray,
    target_obs: np.ndarray,
    asset_index: int,
    feature_names: list[str],
    out_dir: str | Path,
    nsamples: int = 200,
):
    """지정한 시점(target_obs 1건)의 asset_index 비중 결정을 SHAP으로 설명하고
    Summary Plot(배경 표본 전체) + Force Plot(해당 시점)을 저장한다.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predict_fn = _make_scalar_model_fn(sb3_model, asset_index)
    background = shap.sample(background_obs, min(50, len(background_obs)))
    explainer = shap.KernelExplainer(predict_fn, background)

    shap_values_bg = explainer.shap_values(background, nsamples=nsamples)
    plt.figure()
    shap.summary_plot(shap_values_bg, background, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_plot.png", dpi=150)
    plt.close()

    shap_values_target = explainer.shap_values(target_obs.reshape(1, -1), nsamples=nsamples)
    shap.force_plot(
        explainer.expected_value,
        shap_values_target[0],
        target_obs,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.savefig(out_dir / "shap_force_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "summary_plot": str(out_dir / "shap_summary_plot.png"),
        "force_plot": str(out_dir / "shap_force_plot.png"),
        "expected_value": float(explainer.expected_value),
    }
