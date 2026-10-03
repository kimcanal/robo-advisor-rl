"""Walk-Forward 저장물 → artifacts/drl (API가 읽는 서빙 번들).

    python -m rl.export_serving --src rl/outputs/serving_run --reward mdd_penalty

- src/models/{run_tag}에서 해당 보상의 "가장 최근 테스트 윈도우" run들을 복사
- src/monitor/*.monitor.csv 전부 복사 (학습 곡선, 보상 3종 비교용)
- serving.json: run_tags + 스냅샷 관측(로컬 가격 CSV가 있으면 마지막 거래일 기준)

스냅샷은 Docker 안에 가격 CSV가 없어도 /optimize가 "as_of 날짜의 비중"을
돌려줄 수 있게 하기 위한 것이다. 가격 원본은 커밋하지 않는다(yfinance 약관).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .serving import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_SERVING_REWARD,
    build_observation,
    discover_runs,
    load_local_prices,
)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def export(src: Path, dst: Path = DEFAULT_ARTIFACT_DIR, reward: str = DEFAULT_SERVING_REWARD,
           copy_all_rewards: bool = False) -> dict:
    metas = discover_runs(src)
    if not metas:
        raise FileNotFoundError(f"no runs under {src}/models")
    last_test = max(m["test_end"] for m in metas)
    latest = [m for m in metas if m["test_end"] == last_test]
    chosen = [m for m in latest if m["reward_type"] == reward]
    if not chosen:
        raise FileNotFoundError(f"no {reward} runs in latest window (test_end={last_test})")
    to_copy = latest if copy_all_rewards else chosen

    (dst / "models").mkdir(parents=True, exist_ok=True)
    for m in to_copy:
        target = dst / "models" / m["run_tag"]
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(m["run_dir"], target)
    if (src / "monitor").is_dir():
        (dst / "monitor").mkdir(parents=True, exist_ok=True)
        for f in (src / "monitor").glob("*.monitor.csv"):
            shutil.copy2(f, dst / "monitor" / f.name)

    tickers, window = chosen[0]["tickers"], int(chosen[0]["window"])
    snapshot = None
    prices = load_local_prices(tickers)
    if prices is not None:
        obs, as_of = build_observation(prices, tickers, window)
        snapshot = {
            "as_of": str(as_of.date()),
            "obs_raw": [float(x) for x in obs],
            "current_weights": "equal (1/N) — client can override",
            "note": "raw obs derived from local price CSVs at export time",
        }
    cfg = {
        "reward_type": reward,
        "run_tags": [m["run_tag"] for m in chosen],
        "selection_rule": (
            "latest walk-forward window, reward fixed a priori (spec default variant 3, "
            "lambda=1.0); not selected by test performance"
        ),
        "test_window": [chosen[0]["test_start"], chosen[0]["test_end"]],
        "exported_from_commit": _git_sha(),
        "snapshot": snapshot,
    }
    (dst / "serving.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"[export_serving] {len(chosen)} runs → {dst} (reward={reward}, snapshot={'yes' if snapshot else 'no'})")
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default=str(DEFAULT_ARTIFACT_DIR))
    ap.add_argument("--reward", default=DEFAULT_SERVING_REWARD)
    ap.add_argument("--all-rewards", action="store_true",
                    help="최근 윈도우의 보상 3종 모델을 전부 복사 (SHAP 비교용)")
    a = ap.parse_args()
    export(Path(a.src), Path(a.dst), a.reward, a.all_rewards)
