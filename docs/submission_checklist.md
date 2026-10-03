# Submission checklist (morning printout) — Notion week-38

> One-page Done / Yunha-owned / Blocked. Educational demo only — not investment advice.
> See also: `docs/notion_submission_map.md`, `docs/morning_review.md`.

## Done (scaffolding on PR #3)

| Item | Paths |
|---|---|
| Architecture (mermaid) | `docs/architecture.md`, README diagram |
| Reward rationale | `docs/reward_rationale.md`, `rl/rewards.py` |
| Docker / Makefile | `Dockerfile` (3.12), `docker-compose.yml`, `Makefile` (+ `smoke`) |
| Metrics / backtest API | `rl/backtest.py`, `GET /backtest` |
| ANOVA | `rl/stats_tests.py`, `POST /anova`, Streamlit tab |
| RAG LangGraph stub | `rag/graph.py`, `POST /research` (`stub: true`, `node_trace`, `node_latencies_ms`) |
| Risk-tag wiring | `POST /risk-tags/apply`, `rl/risk_tags.py`, Streamlit tab |
| Performance targets sheet | `docs/performance_targets.md` (numbers blank until Colab) |
| Error analysis structure | `docs/error_analysis.md` (H1–H8) |
| ~20p report skeleton | `docs/report/outline.md` |
| Financial disclaimer | README, `GET /health`, Streamlit caption |
| OpenAPI / health contract | `GET /health` → version + `educational` + `endpoints`; `tests/test_openapi_routes.py` |
| CI green (interim pins) | `.github/workflows/ci.yml` + `rl/requirements.txt` env markers — `docs/ci_python_note.md` |

## Yunha-owned (do not invent)

| Item | Notes |
|---|---|
| Colab Walk-Forward numbers | Re-clone **master** (PR#5 KS11 soft-fail); fill `docs/performance_targets.md` from WF CSVs |
| Live RAG / LLM | Keep responses `stub: true` until real keys + store |
| Final PDF export | Paste Colab nums into `docs/report/outline.md` §10 / §13 |

## Blocked (needs token / decision)

| Item | Blocker |
|---|---|
| `ci.yml` Python **3.12** bump | GitHub OAuth lacks `workflow` scope (current: `repo` only). Interim env markers keep GHA green on 3.11. |

## Quick smoke

```bash
source .venv/bin/activate
pytest -q
make api   # then open /docs ; or: make smoke
```

**Do not merge PR #3** until Yunha OK.
