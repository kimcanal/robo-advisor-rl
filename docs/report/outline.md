# Final report outline (PDF skeleton) — 라스트댄스 week-38+

> Educational / research draft. **Not investment advice.** Backtests ≠ future returns.
> Fill each section for Notion → PDF export. Placeholders marked `TODO`.

---

## 1. Title & team

| Field | Value |
|---|---|
| Project | 에이전틱 RAG 및 강화학습 기반 통합 자율 로보어드바이저 |
| Codename | 라스트댄스 |
| Team | 데이터·퀀트 / 강화학습 / 설명·평가 / 리서치·RAG / 백엔드·화면 |
| Disclaimer | Educational demo only — not investment advice |

`TODO`: final author list, presentation date (제안 2026-10-12 / 중간 2026-11-09 / 완료 2026-12-14).

---

## 2. Problem & motivation

- Dual-axis mission: **quantitative** portfolio optimization (RL) + **qualitative** research (RAG).
- Interface: risk tags `{ticker, risk_score, tag, ts}` into `PortfolioEnv.portfolio_risk`.
- Why not a single monolith: parallel team development, clear contracts.

`TODO`: stakeholder pain points, scope boundaries (no live trading by default).

---

## 3. Architecture

```
Data/Quant CSV → RL PortfolioEnv+PPO ← risk_tags ← RAG (plan/retrieve/tag/verify/summarize)
                         ↓
              Explain (SHAP) + ANOVA + Walk-Forward metrics
                         ↓
              FastAPI + Streamlit (+ Docker)
```

`TODO`: insert diagram export from README; list deployed endpoints.

---

## 4. RL design (env, rewards, Safe-Guard)

- Observation: returns window + weights + RSI + MACD + **portfolio_risk**.
- Rewards: simple / sharpe / mdd_penalty (see `rl/README.md`).
- Safe-Guard MDD limit (+ optional `risk_mdd_scale`).
- VecNormalize, Walk-Forward, transaction costs / slippage notes.

`TODO`: key tables (seed-averaged metrics vs equal-weight / MVO).

---

## 5. RAG research agent (Notion-aligned)

- Graph: **plan → retrieve → tag_risk → verify → summarize**.
- Store: in-memory Chroma-lite (`rag/store.py`); swap for ChromaDB later.
- Citations: placeholder quotes (`doc_id`, `source`, `snippet`).
- Env contract: validated tags → RL observation (no look-ahead panel).

`TODO`: real LLM / LangGraph wiring, corpus provenance, Self-Correction examples.

---

## 6. Evaluation (metrics, benchmarks, ANOVA, SHAP)

- 12 metrics via `rl.backtest.compute_metrics` (incl. alpha/beta/IR vs benchmark).
- Benchmarks: S&P500 (SPY) / KOSPI — local CSV preferred; API smoke uses synthetic.
- ANOVA: one-way / two-way educational demos (`POST /anova`).
- SHAP: offline `rl.shap_explain`; API loads JSON artifact if present else clear stub.

`TODO`: final Walk-Forward numbers from `colab_run.ipynb`, error analysis vs equal-weight.

---

## 7. API / UI / Docker / CI

- FastAPI: `/health` `/optimize` `/explain` `/research` `/backtest` `/anova`.
- Streamlit: 7 tabs, API-only (no local model load).
- Docker Compose: api + streamlit.
- CI: `.github/workflows/ci.yml` (pytest).

`TODO`: screenshots of Swagger + Streamlit; latency notes from `/backtest`.

---

## 8. Limitations, ethics, next steps

- Limitations: equal-weight not yet beaten in absolute terms; RAG / SHAP still stubbed for live LLM/KernelExplainer.
- Ethics / disclaimer: educational; no live broker; backtests ≠ future returns.
- Next: real LangGraph+Chroma, offline SHAP artifacts into `/explain`, report PDF polish.

`TODO`: open issues, ownership for each next step, Notion checklist sign-off.
