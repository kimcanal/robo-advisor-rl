"""ANOVA API wrapper — rl.stats_tests on synthetic demo series (no network)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from api.schemas import AnovaRequest, AnovaResponse
from rl.stats_tests import one_way_anova, two_way_anova


def _jsonable(obj: Any) -> Any:
    """Convert statsmodels / pandas artefacts into JSON-friendly values."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, pd.DataFrame):
        reset = obj.reset_index()
        records = reset.to_dict(orient="records")
        return [{str(k): _jsonable(v) for k, v in row.items()} for row in records]
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # statsmodels SimpleTable / summary
    if hasattr(obj, "as_text"):
        return obj.as_text()
    if hasattr(obj, "data") and hasattr(obj, "headers"):
        try:
            return {"headers": list(obj.headers), "data": [list(row) for row in obj.data]}
        except Exception:
            return str(obj)
    return str(obj)


def _demo_one_way_groups(seed: int, n_obs: int) -> dict[str, pd.Series]:
    """Three strategy return series with mild mean shifts (educational demo)."""
    rng = np.random.default_rng(seed)
    return {
        "drl_mdd_penalty": pd.Series(rng.normal(0.0005, 0.01, n_obs)),
        "mvo": pd.Series(rng.normal(0.0003, 0.01, n_obs)),
        "equal_weight": pd.Series(rng.normal(0.0002, 0.01, n_obs)),
    }


def _demo_two_way_frame(seed: int, n_obs: int) -> pd.DataFrame:
    """Long-format daily returns × strategy × regime for two-way ANOVA."""
    rng = np.random.default_rng(seed)
    strategies = ["drl_mdd_penalty", "mvo", "equal_weight"]
    regimes = ["bull", "flat", "bear"]
    rows = []
    per_cell = max(n_obs // (len(strategies) * len(regimes)), 8)
    for strat_i, strat in enumerate(strategies):
        for reg_i, regime in enumerate(regimes):
            mean = 0.0004 - 0.0002 * reg_i + 0.0001 * strat_i
            for _ in range(per_cell):
                rows.append(
                    {
                        "daily_return": float(rng.normal(mean, 0.01)),
                        "strategy": strat,
                        "market_regime": regime,
                    }
                )
    return pd.DataFrame(rows)


def run_anova(req: AnovaRequest) -> AnovaResponse:
    mode = (req.mode or "one_way").lower().strip()
    if mode not in {"one_way", "two_way"}:
        mode = "one_way"

    if mode == "two_way":
        df = _demo_two_way_frame(req.seed, req.n_obs)
        raw = two_way_anova(
            df,
            value_col="daily_return",
            factor1="strategy",
            factor2="market_regime",
        )
        result = {
            "anova_table": _jsonable(raw["anova_table"]),
            "eta_squared": _jsonable(raw["eta_squared"]),
        }
        groups = ["strategy", "market_regime"]
        notes = (
            "Two-way ANOVA on synthetic strategy×regime daily returns via "
            "rl.stats_tests.two_way_anova. Educational only — not investment advice."
        )
    else:
        groups_map = _demo_one_way_groups(req.seed, req.n_obs)
        raw = one_way_anova(groups_map)
        result = {
            "f_stat": _jsonable(raw["f_stat"]),
            "p_value": _jsonable(raw["p_value"]),
            "eta_squared": _jsonable(raw["eta_squared"]),
        }
        if "tukey_hsd" in raw:
            result["tukey_hsd"] = _jsonable(raw["tukey_hsd"])
        groups = list(groups_map.keys())
        notes = (
            "One-way ANOVA on synthetic DRL/MVO/equal-weight return series via "
            "rl.stats_tests.one_way_anova. Educational only — not investment advice."
        )

    return AnovaResponse(
        mode=mode,
        groups=groups,
        n_obs=req.n_obs,
        result=result,
        notes=notes,
        stub=True,
    )
