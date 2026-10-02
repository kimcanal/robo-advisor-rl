# Error analysis — Walk-Forward vs equal-weight (week-38)

> Educational / research notes. **Not investment advice.**  
> **Do not invent Colab metrics.** Fill numeric cells only after Walk-Forward CSVs land.

This page structures the Notion **Error analysis** checklist item. It complements
`rl/README.md` 「알려진 한계 / TODO」and `docs/performance_targets.md`.

---

## 1. What “underperformance” means here

We compare **DRL (PPO + reward × window)** to **equal-weight (EW)** and optionally
**MVO** on the same Walk-Forward test windows, using the same fee/slippage and
risk-free path.

| Comparison | Source of truth | Status |
|---|---|---|
| DRL vs EW (absolute return / Sharpe / MDD) | Colab WF CSVs → `docs/performance_targets.md` | ⏳ pending Colab numbers |
| DRL vs MVO | same | ⏳ |
| DRL vs SPY / KOSPI | `rl/benchmarks.py` (KOSPI soft-fail OK) | ⏳ |

Until Colab numbers exist, treat “EW not yet beaten” as a **working hypothesis
from earlier demos**, not a graded claim.

---

## 2. Hypotheses (why DRL may lag EW)

Ordered for report write-up. Mark each confirmed / rejected only with Colab evidence.

| ID | Hypothesis | Why it could hurt vs EW | How to check (no invented numbers) |
|---|---|---|---|
| H1 | **Missing drawdown in observation** | `mdd_penalty` / Safe-Guard punish drawdown, but agent cannot see current DD → hard to learn avoidance (`rl/README` 한계 6) | Ablation: add DD feature; compare WF Sharpe/MDD vs baseline seed-matched run |
| H2 | **Cost / turnover drag** | Continuous rebalancing + fee/slippage; EW turns over less | Compare average turnover & cost drag columns in WF CSV / backtest extras |
| H3 | **Reward mismatch** | `simple` maximizes episode return; `sharpe` / `mdd_penalty` trade return for stability — may look “worse” on raw return vs EW | Report all 3 rewards; do not crown a winner on one metric |
| H4 | **Train/test regime shift** | WF train≈4y → test≈1y; Bull-trained policy in Bear test | Regime labels (`rl/regime.py`) × method two-way ANOVA from Colab |
| H5 | **VecNormalize train-only stats** | Eval uses frozen train stats; distribution shift on test | Confirm saved stats travel with policy; seed-average gap train vs test |
| H6 | **Hyperparam single-split bias** | `lambda_sweep` / `window_sweep` picked on one split (`rl/README` 한계 7) | Prefer WF multi-window averages over single-split “best” knobs |
| H7 | **Benchmark / data gaps** | KOSPI omitted on yfinance soft-fail; universe/CSV gaps | Note omitted benchmarks explicitly; SPY still required |
| H8 | **Insufficient timesteps / seeds** | Demo timesteps ≪ report-grade ≥100k; single seed noise | Require ≥100k × ≥2 windows × multi-seed before claiming beat/lose EW |

---

## 3. How to read Walk-Forward CSVs (Colab)

Typical artifacts (paths may vary slightly by notebook cell):

| Artifact | Use for error analysis |
|---|---|
| Per-window metrics CSV | DRL vs EW vs MVO per reward / window |
| Incremental WF CSV | Stability across windows; catch one bad window dominating the mean |
| Daily equity / returns (if saved) | Drawdown timing vs EW; turnover spikes |
| ANOVA tables | Method / reward / regime significance (educational) |
| SHAP JSON / plots | Feature attributions — **offline**; do not invent importance ranks |

Suggested read-out order for the PDF:

1. Protocol check: timesteps, rewards, windows, seeds match Notion targets.  
2. Table: mean±std of key metrics (return, Sharpe, Sortino, MDD, turnover) — **paste from CSV**.  
3. Call out windows where DRL ≪ EW (regime? costs?).  
4. Map failures to hypotheses H1–H8; leave untested rows as `TODO`.  
5. Fill `docs/performance_targets.md`; never backfill fake figures.

---

## 4. What Colab will fill (placeholders only)

Copy into the report **only after** the full WF run on **master** (incl. PR#5 KS11 soft-fail):

- [ ] Best reward by Sharpe / by MDD (name only + numbers from CSV)  
- [ ] EW gap (DRL − EW) for return & Sharpe — mean across windows  
- [ ] Whether any reward beats EW on **risk-adjusted** metric even if raw return lags  
- [ ] Regime-conditioned gaps (Bull/Flat/Bear)  
- [ ] Confirmed vs rejected hypothesis IDs  

Owner of numbers: **Yunha (Colab Walk-Forward)**.

---

## 5. Pointers

| Doc / code | Role |
|---|---|
| `rl/README.md` 한계 6–7 | Structural obs / sweep caveats |
| `docs/reward_rationale.md` | Why three rewards exist |
| `docs/performance_targets.md` | Numeric checklist |
| `docs/report/outline.md` §10–13 | Where this lands in the ~20p PDF |
| `rl/walk_forward.py`, `rl/backtest.py` | Code paths producing CSVs |

### Financial disclaimer

교육·연구 목적. 백테스트 ≠ 미래 수익. 투자 자문 아님.
