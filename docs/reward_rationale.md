# Reward rationale (Notion week-38)

> Educational / research notes for the three PPO reward variants.
> **Not investment advice.** Backtests ≠ future returns.
> Detail and citations also live in [`rl/README.md`](../rl/README.md).

---

## Why three rewards?

Course / Notion asks for a clear **why** behind reward design, then statistical
comparison (ANOVA) across variants. We keep three named functions in
`rl/rewards.py` with the same signature so Walk-Forward and demos can swap them
without changing the env loop.

| Name | Formula (sketch) | Role |
|---|---|---|
| `simple` | \(r_t\) | Control — raw step return, no risk term |
| `sharpe` | \(r_t / \sigma_t\) (rolling vol) | Risk-adjusted — penalize high volatility |
| `mdd_penalty` | \(r_t - \lambda \cdot \mathrm{DD}_t\) | Tail / path risk — penalize current drawdown |

Defaults: \(\lambda =\) `MDD_LAMBDA_DEFAULT` (1.0 in `rl/config.py`; sweep range often 0.5–5.0).
See `rl/README.md` design table for Sharpe / Calmar / FinRL-style context (cite primary sources in the PDF, do not invent numbers here).

---

## Safe-Guard MDD

Independent of the reward scalar, `PortfolioEnv` can **terminate** an episode when
portfolio drawdown exceeds `mdd_limit` (default **15%**, `MDD_LIMIT` in
`rl/config.py`). Optional risk-tag scaling may tighten the effective limit via
`portfolio_risk` — that is a hard stop, not a soft reward term.

- Unit coverage: `rl/tests/test_env.py` (Safe-Guard trigger).
- Rationale: hard loss-budget style constraint for educational demos; not a live stop-loss broker hook.

---

## VecNormalize

Training wraps the env with Stable-Baselines3 **VecNormalize** so observation
(and optionally reward) scale is estimated on **train splits only**, then frozen
for eval / Walk-Forward test windows. Saved stats must travel with the policy
zip (`rl/train.py` prints the vecnorm path).

- Why: reduce look-ahead from fitting scalers on the full series; stabilize PPO.
- TODO: attach Colab Walk-Forward paths / seed counts when available — **do not invent metrics**.

---

## How we will compare (placeholders)

| Comparison | Method | Status |
|---|---|---|
| Reward A vs B vs C | One-way ANOVA (+ Tukey if significant) | Code: `rl/stats_tests.py`, API `POST /anova` (synthetic teaching series) |
| vs equal-weight / MVO | Metrics table from `rl.backtest.compute_metrics` | `TODO` Colab numbers → `docs/performance_targets.md` |
| Across regimes | Two-way ANOVA (method × Bull/Flat/Bear) | Code present; `TODO` fill from Colab WF |

`TODO (Colab):` seed-averaged Sharpe / MDD / Calmar (etc.) for each reward — paste from
`colab_run.ipynb` Walk-Forward CSVs only. Never invent or backfill fake figures.

---

## Pointers

- Implementation: `rl/rewards.py`, `rl/env/portfolio_env.py`, `rl/config.py`
- Training / normalize: `rl/train.py`, `rl/walk_forward.py`
- Narrative + literature table: `rl/README.md` (“설계 근거”, “설계 근거 및 출처”)
- Report skeleton: `docs/report/outline.md`

---

## Financial disclaimer

**본 문서는 교육·연구 목적입니다. 투자 자문이 아니며, 특정 증권의 매수·매도를 권유하지 않습니다.**

- 백테스트·시뮬레이션은 과거 또는 합성 데이터에 기반하며 **미래 수익을 보장하지 않습니다**.
- 거래비용·슬리피지·유동성·세금 등이 단순화되어 있을 수 있습니다.
- API / Docker / `.env.example`는 로컬 실험용이며 라이브 브로커는 기본 비활성입니다.
