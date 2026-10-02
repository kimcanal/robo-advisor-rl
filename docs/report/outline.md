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
- Scope: no live broker by default; API keys / Docker are local-experiment only (see disclaimer).

`TODO`: stakeholder pain points (course Notion charter quotes).

---

## 3. Architecture

```
Data/Quant CSV → RL PortfolioEnv+PPO ← risk_tags ← RAG (plan/retrieve/tag/verify/summarize)
                         ↓
              Explain (SHAP) + ANOVA + Walk-Forward metrics
                         ↓
              FastAPI + Streamlit (+ Docker + CI)
```

### Deployed API endpoints (from README)

| Method | Path | Role |
|---|---|---|
| GET | `/health` | Liveness / version |
| POST | `/optimize` | MVO or equal-weight via `rl.mvo` |
| POST | `/explain` | SHAP artifact JSON or clear stub |
| POST | `/research` | RAG plan→retrieve→tag_risk→verify→summarize stub + citations |
| POST | `/risk-tags/apply` | Causal panel + short `PortfolioEnv` `portfolio_risk` wiring demo |
| GET | `/backtest` | Synthetic metrics + optional synth SPY/KOSPI + `latency_ms` |
| POST | `/anova` | Educational one-way / two-way on synthetic series |

Streamlit: 8 tabs, **API-only** (no local model load). Docker Compose: `api` + `streamlit`.

`TODO`: insert diagram export / screenshots from Swagger + Streamlit.

---

## 4. RL design (env, rewards, Safe-Guard)

- Observation: returns window + weights + RSI + MACD + **portfolio_risk**.
- Rewards: simple / sharpe / mdd_penalty (see `rl/README.md`).
- Safe-Guard MDD limit (+ optional `risk_mdd_scale`).
- VecNormalize, Walk-Forward, transaction costs / slippage notes.

`TODO`: key tables (seed-averaged metrics vs equal-weight / MVO) — **pending Colab numbers**.

---

## 5. RAG research agent (Notion-aligned)

### Graph node names (stub)

1. **plan** — decompose query into retrieval / tagging steps  
2. **retrieve** — in-memory Chroma-lite store (`rag/store.py`)  
3. **tag_risk** — emit `{ticker, risk_score[0,1], tag, ts}`  
4. **verify** — schema / citation checks (`verify_ok`, `verify_notes`)  
5. **summarize** — report excerpt + placeholders  

- Store: in-memory Chroma-lite; swap for ChromaDB later.
- Citations: placeholder quotes (`doc_id`, `source`, `snippet`, `quote`).
- Env contract: validated tags → causal `risk_score_panel` → RL observation (no look-ahead).
- Wiring demo: `POST /risk-tags/apply` (mock or rag_graph source; optional short env steps).

`TODO`: real LLM / LangGraph wiring, corpus provenance, Self-Correction examples.

---

## 6. Evaluation (metrics, benchmarks, ANOVA, SHAP)

- 12 metrics via `rl.backtest.compute_metrics` (incl. alpha/beta/IR vs benchmark).
- Benchmarks: S&P500 (SPY) / KOSPI — local CSV preferred; API smoke uses synthetic.
- ANOVA: one-way / two-way educational demos (`POST /anova`).
- SHAP: offline `rl.shap_explain`; API loads JSON artifact if present else clear stub.
- Tracking sheet: `docs/performance_targets.md` (placeholders only until Colab).

`TODO`: final Walk-Forward numbers from `colab_run.ipynb` (≥100k×3 rewards×≥2 windows), error analysis vs equal-weight — **pending Colab numbers**.

---

## 7. API / UI / Docker / CI

- FastAPI endpoints: see §3 table.
- Streamlit tabs: Overview, Optimize, Explain, Research, Risk-tag wiring, Backtest, ANOVA, Health.
- Docker Compose: api + streamlit.
- CI: `.github/workflows/ci.yml` (pytest).

`TODO`: screenshots of Swagger + Streamlit; latency notes from `/backtest`.

---

## 8. Financial disclaimer (copy into PDF)

**본 저장소 / 보고서는 교육·연구 목적의 데모입니다. 투자 자문이 아니며, 특정 증권의 매수·매도를 권유하지 않습니다.**

- 백테스트·시뮬레이션 성과는 과거 데이터(또는 합성 데이터)에 기반하며 **미래 수익을 보장하지 않습니다** (backtests ≠ future returns).
- 거래비용·슬리피지·유동성·세금·survivorship 등 실전 제약이 단순화되어 있을 수 있습니다.
- API 키(`.env.example`)와 Docker 스택은 로컬 실험용이며, 라이브 브로커 연동은 기본 비활성입니다.
- 실제 투자 결정은 본인 책임이며, 필요 시 자격 있는 전문가와 상담하세요.

---

## 9. Limitations, ethics, next steps

- Limitations: equal-weight not yet beaten in absolute terms; RAG / SHAP still stubbed for live LLM/KernelExplainer; performance table pending Colab.
- Ethics / disclaimer: educational; no live broker; backtests ≠ future returns (see §8).
- Next: real LangGraph+Chroma, offline SHAP artifacts into `/explain`, fill `docs/performance_targets.md` from Colab, report PDF polish.

`TODO`: open issues, ownership for each next step, Notion checklist sign-off.
