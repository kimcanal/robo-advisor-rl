"""Read-only access to trained-policy and results artifacts (no training in the API).

Locations (overridable by env for Docker volumes / tests):
    DRL_ARTIFACT_DIR   artifacts/drl      models, serving.json, monitor, shap, lambda sweep
    RESULTS_DIR        docs/results       walk-forward CSV, anova_results.json, figures
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def drl_dir() -> Path:
    return Path(os.getenv("DRL_ARTIFACT_DIR", str(REPO_ROOT / "artifacts" / "drl")))


def results_dir() -> Path:
    return Path(os.getenv("RESULTS_DIR", str(REPO_ROOT / "docs" / "results")))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@lru_cache(maxsize=4)
def _server_for(root: str):
    from rl.serving import DRLServer

    return DRLServer.from_artifacts(Path(root))


def drl_server():
    """Cached DRLServer per artifact dir (models load lazily on first predict)."""
    return _server_for(str(drl_dir()))


def reset_cache() -> None:
    _server_for.cache_clear()


def artifact_status() -> dict:
    s = drl_server().status()
    shap_json = drl_dir() / "shap" / "shap_result.json"
    s.update(
        {
            "shap_main_model": shap_json.is_file(),
            "learning_curves": (drl_dir() / "learning_curves.json").is_file(),
            "lambda_sweep": (drl_dir() / "lambda_sweep.csv").is_file(),
            "walk_forward_results": (results_dir() / "wf_summary.json").is_file(),
            "anova_results": (results_dir() / "anova_results.json").is_file(),
            "rag": "stub (no news corpus ingested)",
        }
    )
    return s


ALLOWED_FILE_ROOTS = {"drl": drl_dir, "results": results_dir}


def safe_file(root_key: str, rel: str) -> Path | None:
    """Resolve a PNG/CSV/JSON under an allowed root; reject traversal."""
    if root_key not in ALLOWED_FILE_ROOTS:
        return None
    root = ALLOWED_FILE_ROOTS[root_key]().resolve()
    p = (root / rel).resolve()
    if root not in p.parents or not p.is_file():
        return None
    if p.suffix.lower() not in {".png", ".csv", ".json"}:
        return None
    return p
