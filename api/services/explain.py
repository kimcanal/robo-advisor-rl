"""Decision explanation.

Order:
  1) Main-model SHAP (artifacts/drl/shap/shap_result.json from rl.shap_main) — real values
  2) Legacy precomputed JSON (rl/outputs/shap/*.json or EXPLAIN_ARTIFACT_PATH)
  3) Unavailable: no contributions are returned (no pseudo / random values)

KernelExplainer is far too slow for a synchronous request, so SHAP is computed
offline and this endpoint only serves it (latency well under the 5 s budget).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from api.schemas import ExplainRequest, ExplainResponse, FeatureContribution
from api.services.artifacts import drl_dir, read_json

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ARTIFACT_DIR = REPO_ROOT / "rl" / "outputs" / "shap"


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)


def _main_model(req: ExplainRequest, t0: float) -> ExplainResponse | None:
    res = read_json(drl_dir() / "shap" / "shap_result.json")
    if not res or not res.get("decisions"):
        return None
    decisions = res["decisions"]
    pick = None
    if req.decision:
        pick = next((d for d in decisions if d["label"] == req.decision), None)
    if pick is None:
        pick = decisions[0]
    contribs = pick["contributions"][: req.top_k]
    plots = {"summary": f"/artifacts/file?root=drl&path=shap/{res['summary']['plot']}"}
    for d in decisions:
        plots[f"force_{d['label']}"] = f"/artifacts/file?root=drl&path=shap/{d['force_plot']}"
    return ExplainResponse(
        asset=pick["asset"],
        asset_index=int(pick["asset_index"]),
        top_features=[
            FeatureContribution(feature=c["feature"], contribution=round(float(c["shap"]), 6))
            for c in contribs
        ],
        summary=(
            f"SHAP for the {pick['asset']} weight decided as of {pick['as_of']} ({pick['label']}), "
            f"model {res['run_tag']} (reward={res['reward_type']}, test {res['test'][0]}~{res['test'][1]}). "
            f"base={pick['base_value']:.4f}, weight={pick['prediction']:.4f}. "
            f"{res['method']}. {res.get('note', '')}"
        ),
        stub=False,
        mode="main_model_shap",
        as_of=pick["as_of"],
        base_value=float(pick["base_value"]),
        prediction=float(pick["prediction"]),
        run_tag=res["run_tag"],
        plots=plots,
        decisions=[d["label"] for d in decisions],
        global_importance=res["summary"]["mean_abs_shap"][: max(req.top_k, 10)],
        latency_ms=_ms(t0),
    )


def _legacy_artifact(req: ExplainRequest) -> Path | None:
    candidates: list[Path] = []
    if req.artifact_path:
        candidates.append(Path(req.artifact_path))
    if os.getenv("EXPLAIN_ARTIFACT_PATH"):
        candidates.append(Path(os.environ["EXPLAIN_ARTIFACT_PATH"]))
    if LEGACY_ARTIFACT_DIR.is_dir():
        candidates.extend(sorted(LEGACY_ARTIFACT_DIR.glob("*.json")))
    return next((p for p in candidates if p.is_file()), None)


def _load_legacy(path: Path, asset: str, top_k: int) -> list[FeatureContribution] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
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
    out = []
    for row in (rows or [])[:top_k]:
        if not isinstance(row, dict):
            continue
        feat = row.get("feature") or row.get("name")
        contrib = row.get("contribution", row.get("shap", row.get("value")))
        if feat is None or contrib is None:
            continue
        out.append(FeatureContribution(feature=str(feat), contribution=round(float(contrib), 6)))
    return out or None


def run_explain(req: ExplainRequest) -> ExplainResponse:
    t0 = time.perf_counter()
    main = _main_model(req, t0)
    if main is not None:
        return main

    tickers = list(req.tickers) or ["SPY"]
    idx = min(req.asset_index, len(tickers) - 1)
    asset = tickers[idx]
    path = _legacy_artifact(req)
    if path is not None:
        loaded = _load_legacy(path, asset, req.top_k)
        if loaded:
            return ExplainResponse(
                asset=asset, asset_index=idx, top_features=loaded, stub=False,
                mode="artifact_json", latency_ms=_ms(t0),
                summary=f"Loaded precomputed SHAP contributions for {asset} from {path}.",
            )
    return ExplainResponse(
        asset=asset, asset_index=idx, top_features=[], stub=True, mode="unavailable",
        latency_ms=_ms(t0),
        summary=(
            "No SHAP for the trained policy yet: artifacts/drl/shap/shap_result.json is missing. "
            "Run `python -m rl.shap_main` after exporting the serving models (Colab cell 7). "
            "No contribution values are returned rather than placeholders."
        ),
    )
