"""Read-only results for the dashboard: ANOVA, learning/lambda curves, portfolio history."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.schemas import ArtifactPayload
from api.services.artifacts import drl_dir, drl_server, read_json, results_dir


def anova_results() -> ArtifactPayload:
    p = results_dir() / "anova_results.json"
    data = read_json(p)
    if data is None:
        return ArtifactPayload(available=False, source=str(p),
                               reason="run python -m rl.analysis.wf_analysis")
    return ArtifactPayload(available=True, source=str(p), data=data)


def training_curves() -> ArtifactPayload:
    lc = read_json(drl_dir() / "learning_curves.json")
    lam_path = drl_dir() / "lambda_sweep.csv"
    lam = None
    if lam_path.is_file():
        df = pd.read_csv(lam_path)
        cols = [c for c in ("lambda", "seed", "cumulative_return", "mdd", "sharpe", "calmar",
                            "evaluated_days", "policy_days", "safe_guard_date") if c in df.columns]
        lam = df[cols].replace({np.nan: None}).to_dict(orient="records")
    if lc is None and lam is None:
        return ArtifactPayload(
            available=False, source=str(drl_dir()),
            reason=("learning_curves.json / lambda_sweep.csv not produced yet "
                    "(Colab cells 6 and 8)"),
        )
    data = {"learning": lc, "lambda_sweep": lam,
            "plots": {k: f"/artifacts/file?root=drl&path={v}" for k, v in
                      (("learning", "learning_curves.png"), ("lambda", "lambda_tradeoff.png"))
                      if (drl_dir() / v).is_file()}}
    return ArtifactPayload(available=True, source=str(drl_dir()), data=data)


def portfolio_history() -> ArtifactPayload:
    """Serving model's walk-forward test year: cum. return, drawdown, held weights, VaR/CVaR."""
    server = drl_server()
    if not server.available:
        return ArtifactPayload(available=False, source=str(drl_dir()),
                               reason="no trained DRL policy under artifacts/drl")
    series, weights = [], []
    for run in server.runs:
        p = run.run_dir / "test_daily.csv"
        if p.is_file():
            df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
            series.append(df["daily_return"].rename(f"seed{run.meta.get('seed')}"))
            weights.append(df[[c for c in df.columns if c.startswith("w_")]])
    if not series:
        return ArtifactPayload(available=False, source=str(drl_dir()), reason="test_daily.csv missing")
    rets = pd.concat(series, axis=1)
    ens = rets.mean(axis=1)  # seed-ensemble approximation (mean of log returns)
    cum = np.exp(ens.cumsum())
    dd = (cum.cummax() - cum) / cum.cummax()
    w_mean = sum(weights) / len(weights)
    var95 = float(-np.percentile(ens, 5))
    cvar95 = float(-ens[ens <= np.percentile(ens, 5)].mean())
    safe_guard = [
        {"seed": r.meta.get("seed"), "date": r.meta.get("safe_guard_date")}
        for r in server.runs if r.meta.get("safe_guard_date")
    ]
    data = {
        "dates": [str(d.date()) for d in ens.index],
        "cumulative": [float(x) for x in cum],
        "drawdown": [float(x) for x in dd],
        "per_seed_cumulative": {c: [float(x) for x in np.exp(rets[c].cumsum())] for c in rets},
        "last_weights": {c[2:]: float(v) for c, v in w_mean.iloc[-1].items()},
        "weights_monthly": w_mean.resample("ME").last().reset_index()
        .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "var_95": var95,
        "cvar_95": cvar95,
        "mdd": float(dd.max()),
        "mdd_limit": 0.15,
        "safe_guard_events": safe_guard,
        "status": server.status(),
        "note": ("Walk-forward TEST year of the serving model(s); curve = mean of seed daily "
                 "log returns. After a Safe-Guard trigger the run holds cash (w_CASH=1)."),
    }
    return ArtifactPayload(available=True, source=str(drl_dir()), data=data)
