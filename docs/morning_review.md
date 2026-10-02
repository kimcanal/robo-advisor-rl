# Morning review checklist (Yunha) — 2026-10-03

> Left in-repo by overnight bot. **PR #3 stays OPEN** until you OK a merge.
> Educational demo only — not investment advice.

## Do first

1. Open [PR #3](https://github.com/kimcanal/robo-advisor-rl/pull/3) — tip should be `ae27e28` or later; **CI green**.
2. Skim diff: API/Streamlit/Docker/RAG stubs + docs only; RL training core unchanged aside from earlier KS11 soft-fail on master.
3. Optional: `make compose-up` or `make api` + Swagger `/docs`; with API up, `make smoke` hits `/health`.
4. One-page printout: `docs/submission_checklist.md` (Done / Yunha-owned / Blocked).

## Overnight delta (~06:30 KST)

- OpenAPI / route contract tests (`tests/test_openapi_routes.py`) — required paths + `/health` educational/disclaimer/version + `/research` stub trace.
- `GET /health` now returns `endpoints` list; OpenAPI tag descriptions; API **0.1.5**.
- Streamlit Health tab shows API version + educational disclaimer from `/health`.
- `docs/submission_checklist.md` — morning one-pager; linked from README + Notion map.
- Tip SHA synced to `ae27e28` (was stale `0f64419`).

## CI note

- Preferred fix was `.github/workflows/ci.yml` → Python **3.12**, but push of workflow files needs GitHub `workflow` OAuth scope (current token: `repo` only).
- Interim: env markers in `rl/requirements.txt` (Colab ≥3.12 keeps exact pins). Details: `docs/ci_python_note.md`.
- Optional later: add `workflow` scope, bump ci.yml to 3.12, drop `<3.12` marker lines.

## Your Colab ownership

1. Re-clone **master** (includes PR#5 KS11 soft-fail) before full Walk-Forward.
2. Fill `docs/performance_targets.md` from WF CSVs — **do not invent numbers**.
3. Use `docs/error_analysis.md` H1–H8 when writing Why/How for EW gaps.
4. Paste into PDF via `docs/report/outline.md` §10 / §13.

## Notion checklist (PR #3)

Mostly ✅ (`docs/notion_submission_map.md`, `docs/submission_checklist.md`). Still pending on **your** side: Colab WF numbers, live RAG/LLM, final PDF export.

## Do not

- Do not expect overnight bot to merge PR #3.
- Do not paste invented Sharpe/MDD into README or the report.
