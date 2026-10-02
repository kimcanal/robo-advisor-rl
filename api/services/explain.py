"""Decision explanation — SHAP artifact path if present, else clear stub.

Avoid importing rl.shap_explain at module load (it pulls shap/matplotlib).
Feature-name layout mirrors PortfolioEnv observation order.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from api.schemas import ExplainRequest, ExplainResponse, FeatureContribution

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = REPO_ROOT / "rl" / "outputs" / "models"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "rl" / "outputs" / "shap"


def _feature_names(tickers: list[str], window: int = 30, *, include_risk: bool = True) -> list[str]:
    names: list[str] = []
    for k in range(window, 0, -1):
        for t in tickers:
            names.append(f"ret_t-{k}_{t}")
    names += [f"weight_{t}" for t in tickers]
    names += [f"rsi_{t}" for t in tickers]
    names += [f"macd_{t}" for t in tickers]
    if include_risk:
        names.append("portfolio_risk")
    return names


def _resolve_artifact_path(req: ExplainRequest) -> Path | None:
    candidates: list[Path] = []
    if req.artifact_path:
        candidates.append(Path(req.artifact_path))
    env_path = os.getenv("EXPLAIN_ARTIFACT_PATH")
    if env_path:
        candidates.append(Path(env_path))
    if DEFAULT_ARTIFACT_DIR.is_dir():
        candidates.extend(sorted(DEFAULT_ARTIFACT_DIR.glob("*.json")))
    for p in candidates:
        if p.is_file():
            return p
    return None


def _load_artifact(path: Path, asset: str, top_k: int) -> list[FeatureContribution] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # Accept {"features":[{"feature","contribution"}, ...]} or per-asset map
    rows = None
    if isinstance(data, dict):
        if asset in data and isinstance(data[asset], list):
            rows = data[asset]
        elif "features" in data:
            rows = data["features"]
        elif "top_features" in data:
            rows = data["top_features"]
    elif isinstance(data, list):
        rows = data
    if not rows:
        return None
    out: list[FeatureContribution] = []
    for row in rows[:top_k]:
        if not isinstance(row, dict):
            continue
        feat = row.get("feature") or row.get("name")
        contrib = row.get("contribution", row.get("shap", row.get("value")))
        if feat is None or contrib is None:
            continue
        out.append(FeatureContribution(feature=str(feat), contribution=round(float(contrib), 6)))
    return out or None


def _model_zip_exists(req: ExplainRequest) -> Path | None:
    if req.model_path:
        p = Path(req.model_path)
        if p.is_file():
            return p
    env_path = os.getenv("EXPLAIN_MODEL_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    if DEFAULT_MODELS_DIR.is_dir():
        zips = sorted(DEFAULT_MODELS_DIR.glob("*.zip"))
        if zips:
            return zips[0]
    return None


def run_explain(req: ExplainRequest) -> ExplainResponse:
    t0 = time.perf_counter()
    tickers = list(req.tickers) or ["SPY"]
    idx = min(req.asset_index, len(tickers) - 1)
    asset = tickers[idx]

    # 1) Prefer precomputed SHAP JSON artifacts (CI-friendly, no SB3 load).
    artifact = _resolve_artifact_path(req)
    if artifact is not None:
        loaded = _load_artifact(artifact, asset, req.top_k)
        if loaded:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return ExplainResponse(
                asset=asset,
                asset_index=idx,
                top_features=loaded,
                summary=(
                    f"Loaded precomputed SHAP contributions for {asset} from "
                    f"{artifact}. Replace artifact or drop file to fall back to stub."
                ),
                stub=False,
                mode="artifact_json",
                latency_ms=latency_ms,
            )

    # 2) Model zip present → note that KernelExplainer path is available but
    #    not executed in the API (too slow / heavy for request path). Clear stub.
    model_zip = _model_zip_exists(req)
    names = _feature_names(tickers, window=30, include_risk=True)
    rng = np.random.default_rng(abs(hash(asset)) % (2**32))
    scores = rng.normal(0, 1, size=len(names))
    order = np.argsort(-np.abs(scores))[: req.top_k]
    top = [
        FeatureContribution(feature=names[i], contribution=round(float(scores[i]), 4))
        for i in order
    ]

    if model_zip is not None:
        mode = "shap_kernel_attempted"
        summary = (
            f"Found SB3 zip at {model_zip} but API uses a lightweight pseudo-SHAP "
            f"ranking for {asset} (index {idx}) — KernelExplainer is too slow for "
            "sync HTTP. Run rl.shap_explain.explain_decision offline and drop JSON "
            f"under {DEFAULT_ARTIFACT_DIR}/ or set EXPLAIN_ARTIFACT_PATH."
        )
        stub = True
    else:
        mode = "stub_pseudo_shap"
        summary = (
            f"Stub explanation for {asset} weight (index {idx}). No model zip under "
            f"{DEFAULT_MODELS_DIR} and no SHAP JSON under {DEFAULT_ARTIFACT_DIR}. "
            "Wire offline rl.shap_explain output or set EXPLAIN_ARTIFACT_PATH."
        )
        stub = True

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return ExplainResponse(
        asset=asset,
        asset_index=idx,
        top_features=top,
        summary=summary,
        stub=stub,
        mode=mode,
        latency_ms=latency_ms,
    )
