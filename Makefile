# Educational convenience targets — not for production trading.
.PHONY: help test api streamlit compose-up compose-down fmt-check

help:
	@echo "Targets: test | api | streamlit | compose-up | compose-down | fmt-check"
	@echo "Docs: docs/morning_review.md docs/notion_submission_map.md docs/error_analysis.md"

PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest
UVICORN ?= .venv/bin/uvicorn
STREAMLIT ?= .venv/bin/streamlit

test:
	PYTHONPATH=. $(PYTEST) rl/tests tests/ -q

api:
	PYTHONPATH=. $(UVICORN) api.main:app --reload --host 127.0.0.1 --port 8000

streamlit:
	PYTHONPATH=. API_BASE_URL=http://127.0.0.1:8000 $(STREAMLIT) run streamlit_app/app.py

compose-up:
	docker compose up --build

compose-down:
	docker compose down

fmt-check:
	@if command -v ruff >/dev/null 2>&1; then ruff check api streamlit_app rag tests; \
	elif [ -x .venv/bin/ruff ]; then .venv/bin/ruff check api streamlit_app rag tests; \
	else echo "ruff not installed — skip"; fi
	@if command -v flake8 >/dev/null 2>&1; then flake8 api streamlit_app rag tests; \
	elif [ -x .venv/bin/flake8 ]; then .venv/bin/flake8 api streamlit_app rag tests; \
	else echo "flake8 not installed — skip"; fi
