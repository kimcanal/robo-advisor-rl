# Notion week-38 → repo map

Quick index for graders / morning review. Educational demo only.

| Notion section | Status | Primary paths |
|---|---|---|
| Architecture | ✅ | `docs/architecture.md` (mermaid); `README.md` diagram; `docs/report/outline.md` §4 |
| Reward rationale | ✅ | `docs/reward_rationale.md`, `rl/README.md`, `rl/rewards.py` |
| Docker placeholder | ✅ | `Dockerfile` (3.12), `docker-compose.yml`, `Makefile` |
| Metrics | ✅ code / ⏳ Colab nums | `rl/backtest.py`, `GET /backtest`, `docs/performance_targets.md` |
| ANOVA | ✅ | `rl/stats_tests.py`, `POST /anova`, Streamlit tab |
| RAG / LangGraph | ✅ stub | `rag/graph.py`, `rag/store.py`, `POST /research` (`stub: true`) |
| Risk-tag wiring | ✅ stub | `POST /risk-tags/apply`, `rl/risk_tags.py`, Streamlit tab |
| Performance targets | ✅ sheet / ⏳ nums | `docs/performance_targets.md` |
| CI | ✅ green | `.github/workflows/ci.yml` + `docs/ci_python_note.md` |
| Error analysis | ✅ structure / ⏳ nums | `docs/error_analysis.md` |
| Financial disclaimer | ✅ | README, API `/health`, Streamlit caption, outline §12 |
| ~20p PDF report | ✅ skeleton | `docs/report/outline.md` (+ App D) |
| Colab WF numbers | ⏳ Yunha | `colab_run.ipynb` on **master** |

See also: `docs/morning_review.md`.
