# Final report outline (PDF skeleton) — 라스트댄스 week-38+

> Educational / research draft. **Not investment advice.** Backtests ≠ future returns.
> Fill each section for Notion → PDF export (~20 pages). Placeholders marked `TODO`.
> **Never invent Colab / SHAP / ANOVA live metrics** — paste from Colab CSVs only.

---

## 1. Title & team (~0.5–1 p)

| Field | Value |
|---|---|
| Project | 에이전틱 RAG 및 강화학습 기반 통합 자율 로보어드바이저 |
| Codename | 라스트댄스 |
| Team | 데이터·퀀트 / 강화학습 / 설명·평가 / 리서치·RAG / 백엔드·화면 |
| Disclaimer | Educational demo only — not investment advice |

`TODO`: final author list, presentation date (제안 2026-10-12 / 중간 2026-11-09 / 완료 2026-12-14).

---

## 2. Problem & motivation (~1–1.5 p)

- Dual-axis mission: **quantitative** portfolio optimization (RL) + **qualitative** research (RAG).
- Interface: risk tags `{ticker, risk_score, tag, ts}` into `PortfolioEnv.portfolio_risk`.
- Why not a single monolith: parallel team development, clear contracts.
- Scope: no live broker by default; API keys / Docker are local-experiment only (see disclaimer).

`TODO`: stakeholder pain points (course Notion charter quotes).

---

## 3. Related work (~1.5–2 p)

| Theme | Pointer (cite primary sources in PDF) | Repo note |
|---|---|---|
| PPO | Schulman et al., 2017 | Algorithm choice for continuous weights |
| FinRL / trading Gym | Liu et al., 2020 | Custom env + PPO baseline pattern |
| Sharpe / downside / CVaR | Sharpe 1966; Sortino 1991; Rockafellar & Uryasev 2000 | Metrics & reward inspiration |
| Markowitz MVO | Markowitz, 1952 | Static baseline vs DRL |
| SHAP | Lundberg & Lee, 2017 | Offline explain artifacts |
| ANOVA / Tukey | Fisher; Tukey | Reward / method comparison |
| Regime labels | Dynamic-Regime-Portfolio (MIT) inspiration | `rl/regime.py` (core conditions only) |

Full design table: `rl/README.md`. Reward narrative: `docs/reward_rationale.md`.

`TODO`: 8–12 formal bibliography entries for the PDF.

---

## 4. Architecture (~1.5–2 p)

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
| GET | `/health` | Liveness / version / educational disclaimer |
| POST | `/optimize` | MVO or equal-weight via `rl.mvo` |
| POST | `/explain` | SHAP artifact JSON or clear stub |
| POST | `/research` | RAG plan→retrieve→tag_risk→verify→summarize stub + citations |
| POST | `/risk-tags/apply` | Causal panel + short `PortfolioEnv` `portfolio_risk` wiring demo |
| GET | `/backtest` | Synthetic metrics + optional synth SPY/KOSPI + `latency_ms` |
| POST | `/anova` | Educational one-way / two-way on synthetic series |

Streamlit: 8 tabs, **API-only** (no local model load). Docker Compose: `api` (+ healthcheck) + `streamlit` (depends on healthy api).

`TODO`: insert diagram export / screenshots from Swagger + Streamlit.

---

## 5. Data & features (~1.5–2 p)

- Universe (demo): ETF set via `rl/data/fetch_real_data.py` (SPY/QQQ/…/VNQ); loader prefers `data/raw/{ticker}.csv`, else dummy GBM.
- Risk-free: BIL / short-rate path (`rl/riskfree.py`) — not a hardcoded 0.
- Features: log returns, RSI(14), MACD(12,26,9), current weights, **portfolio_risk**.
- Benchmarks: SPY required for WF smoke; KOSPI soft-fail (`^KS11` → `KS11` → `EWY`, omit if download fails — PR#5).
- Costs: fee / slippage per course spec (`rl/config.py`).

`TODO`: date ranges, missing-data handling notes, survivorship caveat from Colab Data cell.

---

## 6. RL design (env, rewards, Safe-Guard) (~2 p)

- Observation: returns window + weights + RSI + MACD + **portfolio_risk**.
- Action: continuous → softmax weights (long-only, sum=1).
- Rewards: simple / sharpe / mdd_penalty — see `docs/reward_rationale.md` + `rl/README.md`.
- Safe-Guard MDD limit (+ optional `risk_mdd_scale`).
- VecNormalize on train only; Walk-Forward retrain windows.
- Transaction costs / turnover notes.

`TODO`: key tables (seed-averaged metrics vs equal-weight / MVO) — **pending Colab numbers**.

---

## 7. Risk-tag interface contract (~1–1.5 p)

Schema (RAG → RL):

```json
{ "ticker": "SPY", "risk_score": 0.0-1.0, "tag": "string", "ts": "YYYY-MM-DD" }
```

- Causality: tags mapped into a `risk_score_panel` aligned to dates **without look-ahead** (`rl/risk_tags.py`).
- Env: weighted average → observation channel `portfolio_risk` (0 if no tags).
- Demo API: `POST /risk-tags/apply` (mock / client / rag_graph stub + optional short env steps).
- Streamlit: **Risk-tag wiring** tab (API-only).

`TODO`: production corpus provenance; live LLM Self-Correction examples.

---

## 8. RAG research agent (Notion-aligned) (~1.5 p)

### Graph node names (stub)

1. **plan** — decompose query into retrieval / tagging steps  
2. **retrieve** — in-memory Chroma-lite store (`rag/store.py`)  
3. **tag_risk** — emit `{ticker, risk_score[0,1], tag, ts}`  
4. **verify** — schema / citation checks (`verify_ok`, `verify_notes`)  
5. **summarize** — report excerpt + placeholders  

- Store: in-memory Chroma-lite; swap for ChromaDB later.
- Citations: placeholder quotes (`doc_id`, `source`, `snippet`, `quote`).
- Env contract: validated tags → causal panel → RL observation.

`TODO`: real LLM / LangGraph wiring, corpus provenance, Self-Correction examples.

---

## 9. Experiment protocol (~1.5–2 p)

| Knob | Spec / intent | Where |
|---|---|---|
| Timesteps | ≥100k for report-grade WF | `colab_run.ipynb`, `rl/walk_forward.py` |
| Rewards | all 3 | `simple`, `sharpe`, `mdd_penalty` |
| Windows | ≥2 WF windows (train≈4y → test≈1y pattern) | walk-forward helper |
| Seeds | multi-seed average | `TODO` Colab |
| Baselines | equal-weight, MVO | `rl/mvo.py`, metrics |
| Benchmarks | SPY; KOSPI best-effort | `rl/benchmarks.py` (soft-fail KOSPI) |
| Lambda / window sweeps | educational demos | `rl/experiments.py` |

`TODO`: exact seed list, wall-clock, hardware (Colab GPU/CPU), artifact paths.

---

## 10. Evaluation (metrics, benchmarks, ANOVA, SHAP) (~2–2.5 p)

- 12 metrics via `rl.backtest.compute_metrics` (incl. alpha/beta/IR vs benchmark).
- Benchmarks: S&P500 (SPY) / KOSPI — local CSV preferred; API smoke uses synthetic; Colab may omit KOSPI if download soft-fails.
- ANOVA: one-way / two-way educational demos (`POST /anova`).
- SHAP: offline `rl.shap_explain`; API loads JSON artifact if present else clear stub.
- Tracking sheet: `docs/performance_targets.md` (placeholders only until Colab).

`TODO`: final Walk-Forward numbers from `colab_run.ipynb` (≥100k×3 rewards×≥2 windows), error analysis vs equal-weight — **pending Colab numbers. Do not invent.**

---

## 11. API / UI / Docker / CI (~1 p)

- FastAPI endpoints: see §4 table.
- Streamlit tabs: Overview, Optimize, Explain, Research, Risk-tag wiring, Backtest, ANOVA, Health.
- Docker Compose: api healthcheck on `/health`; streamlit `depends_on` with `condition: service_healthy`.
- Make targets: `make test|api|streamlit|compose-up|compose-down|fmt-check`.
- CI: `.github/workflows/ci.yml` (pytest).

`TODO`: screenshots of Swagger + Streamlit; latency notes from `/backtest`.

---

## 12. Financial disclaimer (copy into PDF) (~0.5 p)

**본 저장소 / 보고서는 교육·연구 목적의 데모입니다. 투자 자문이 아니며, 특정 증권의 매수·매도를 권유하지 않습니다.**

- 백테스트·시뮬레이션 성과는 과거 데이터(또는 합성 데이터)에 기반하며 **미래 수익을 보장하지 않습니다** (backtests ≠ future returns).
- 거래비용·슬리피지·유동성·세금·survivorship 등 실전 제약이 단순화되어 있을 수 있습니다.
- API 키(`.env.example`)와 Docker 스택은 로컬 실험용이며, 라이브 브로커 연동은 기본 비활성입니다.
- 실제 투자 결정은 본인 책임이며, 필요 시 자격 있는 전문가와 상담하세요.

---

## 13. Limitations & ethics (~1–1.5 p)

- Limitations: equal-weight not yet beaten in absolute terms (pending Colab); RAG / SHAP still stubbed for live LLM/KernelExplainer; performance table pending Colab; KOSPI may be omitted when yfinance fails (soft-fail).
- Ethics: educational only; no live broker; no invented metrics; clear stub labels on API responses (`stub: true`).
- Risk of overclaiming: demos and synthetic ANOVA must be labeled as teaching aids.

`TODO`: open issues, ownership for each next step, Notion checklist sign-off.

---

## 14. Next steps (~0.5 p)

- Real LangGraph + Chroma, offline SHAP artifacts into `/explain`.
- Fill `docs/performance_targets.md` from Colab WF CSVs.
- Report PDF polish from this outline (~20 pages).
- Optional: expand risk-tag corpus and verification traces.

---

## Appendix A — API table (~0.5 p)

Copy §4 endpoint table; note OpenAPI at `/docs` when stack is up.

## Appendix B — Environment variables (~0.5 p)

From `.env.example`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `RAG_COLLECTION`, market-data keys, `EXPLAIN_ARTIFACT_PATH`, `EXPLAIN_MODEL_PATH`, `API_BASE_URL`, `LOG_LEVEL`, `ENABLE_LIVE_TRADING=false`.

## Appendix C — Reproduce steps (~1 p)

```bash
git clone https://github.com/kimcanal/robo-advisor-rl.git
cd robo-advisor-rl
python3 -m venv .venv && source .venv/bin/activate
pip install -r rl/requirements.txt -r requirements-api.txt
make test          # or: pytest rl/tests tests/ -q
make api           # Swagger http://127.0.0.1:8000/docs
make streamlit     # http://127.0.0.1:8501
# optional: make compose-up
```

Colab: open `colab_run.ipynb` from **master** (includes KS11 soft-fail after PR#5); run Setup → Data → Walk-Forward; download CSVs; fill performance targets — **do not invent numbers**.

`TODO`: pin commit SHA used for the graded Colab run.
