# Performance targets checklist (Notion week-38)

> Educational / research tracking sheet. **Not investment advice.**
> Fill numeric cells only from Colab Walk-Forward / demo runs — **do not invent metrics**.
> Status placeholders: `pending Colab numbers` | `stub` | `n/a`.

Owner of final numbers: **Yunha (Colab Walk-Forward)**. This doc is a checklist only.

---

## 1. Evaluation metrics (12) — DRL vs baselines

Source of truth: `rl.backtest.compute_metrics` (and WF comparison tables).

| # | Metric | DRL (best reward) | Equal-weight (EW) | MVO | SPY | KOSPI | Status |
|---|---|---|---|---|---|---|---|
| 1 | cumulative_return | — | — | — | — | — | pending Colab numbers |
| 2 | cagr | — | — | — | — | — | pending Colab numbers |
| 3 | ann_vol | — | — | — | — | — | pending Colab numbers |
| 4 | var_95 | — | — | — | — | — | pending Colab numbers |
| 5 | cvar_95 | — | — | — | — | — | pending Colab numbers |
| 6 | mdd | — | — | — | — | — | pending Colab numbers |
| 7 | sharpe | — | — | — | — | — | pending Colab numbers |
| 8 | sortino | — | — | — | — | — | pending Colab numbers |
| 9 | calmar | — | — | — | — | — | pending Colab numbers |
| 10 | alpha (vs bench) | — | — | — | n/a | n/a | pending Colab numbers |
| 11 | beta (vs bench) | — | — | — | n/a | n/a | pending Colab numbers |
| 12 | information_ratio | — | — | — | n/a | n/a | pending Colab numbers |

Benchmarks: **EW / MVO / SPY / KOSPI** (local CSV preferred; Colab fetches via `fetch_real_data` / yfinance).

API smoke path (`GET /backtest`) uses **synthetic** series only — mark as `stub`, never paste into the table above.

---

## 2. Walk-Forward protocol targets

| Target | Spec | Status |
|---|---|---|
| Timesteps per reward | ≥ **100,000** | pending Colab numbers |
| Reward functions | **3** (`simple` / `sharpe` / `mdd_penalty`) | pending Colab numbers |
| Windows | ≥ **2** Walk-Forward windows | pending Colab numbers |
| VecNormalize | On (as in `walk_forward.py`) | stub (code ready) |
| Incremental CSV | WF writes incremental results | stub (code ready) |
| Notebook | `colab_run.ipynb` (master clone each run) | stub (notebook on master via PR#4) |

---

## 3. ANOVA

| Item | Where | Status |
|---|---|---|
| One-way (DRL vs MVO vs EW) | `rl.stats_tests.one_way_anova` + Colab cell | pending Colab numbers |
| Two-way (strategy × regime) | `rl.stats_tests.two_way_anova` + Colab cell | pending Colab numbers |
| API educational demo | `POST /anova` (synthetic series) | stub |

---

## 4. SHAP / explainability

| Item | Where | Status |
|---|---|---|
| Offline SHAP (Summary / Force) | `rl.shap_explain` / Colab | pending Colab numbers |
| API `/explain` | artifact JSON or clear stub | stub |

---

## 5. Risk-tag interface (Notion gap)

| Item | Where | Status |
|---|---|---|
| Schema `{ticker, risk_score, tag, ts}` | `rl.risk_tags` | stub (code ready) |
| Causal panel (no look-ahead) | `risk_score_panel` | stub (code ready) |
| Env obs `portfolio_risk` | `PortfolioEnv` | stub (code ready) |
| API wiring demo | `POST /risk-tags/apply` | stub |
| Live RAG / LLM tags | LangGraph + Chroma | pending (real corpus / LLM) |

---

## 6. How to fill this sheet

1. Run Colab `colab_run.ipynb` (demo **or** full WF — Yunha owns the full run).
2. Copy metrics from WF / demo output CSVs under `rl/outputs/` (or Colab downloads).
3. Replace `pending Colab numbers` with actual floats; leave API/synth rows as `stub`.
4. Never backfill invented sharpe/MDD/CAGR into README or the PDF report.
5. For **why** DRL may lag EW (hypotheses, CSV read-out), see `docs/error_analysis.md` — fill confirm/reject only after Colab.

**Disclaimer:** Educational demo only. Backtests ≠ future returns. Not investment advice.
