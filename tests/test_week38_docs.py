"""Light checks that week-38 Notion scaffolding files stay present."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_python_312():
    text = (ROOT / "Dockerfile").read_text()
    assert "python:3.12" in text


def test_requirements_env_markers_for_ci():
    text = (ROOT / "rl" / "requirements.txt").read_text()
    assert 'numpy==2.5.3; python_version >= "3.12"' in text
    assert 'numpy==2.4.6; python_version < "3.12"' in text
    assert "scipy==1.18.1" in text
    assert "shap==0.52.0" in text


def test_notion_docs_exist():
    for rel in (
        "docs/error_analysis.md",
        "docs/ci_python_note.md",
        "docs/report/outline.md",
        "docs/performance_targets.md",
        "docs/reward_rationale.md",
        "docs/morning_review.md",
        "docs/notion_submission_map.md",
        "rag/README.md",
        ".env.example",
    ):
        assert (ROOT / rel).is_file(), rel
