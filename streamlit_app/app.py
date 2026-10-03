"""Streamlit dashboard — FastAPI client only (never loads model files).

Tabs follow the mission spec (4-11): 포트폴리오 현황 / 강화학습 성과 / SHAP 해석 /
에이전트 리서치 / ANOVA 검증 결과 / 리스크 모니터링, plus an ops tab.
Every number shown comes from an API response; when an artifact is missing the
API says so and this page shows that message instead of placeholder numbers.
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DISCLAIMER = (
    "교육용 데모입니다. 투자 자문이 아니며, 백테스트 성과는 미래 수익을 보장하지 않습니다. "
    "Educational demo only — not investment advice; backtests do not guarantee future returns."
)

st.set_page_config(page_title="라스트댄스 AUTO ADVISER", layout="wide")
st.title("라스트댄스 — AUTO ADVISER (교육용)")
st.caption(f"API: `{API_BASE}` · 대시보드는 모델 파일을 직접 읽지 않고 API만 호출합니다.")


def _get(path: str, **params):
    with httpx.Client(timeout=30.0) as c:
        r = c.get(f"{API_BASE}{path}", params=params or None)
        r.raise_for_status()
        return r.json()


def _post(path: str, payload: dict):
    with httpx.Client(timeout=60.0) as c:
        r = c.post(f"{API_BASE}{path}", json=payload)
        return r.status_code, r.json()


def _image(api_path: str, caption: str = ""):
    try:
        with httpx.Client(timeout=30.0) as c:
            r = c.get(f"{API_BASE}{api_path}")
        if r.status_code == 200:
            st.image(r.content, caption=caption, use_container_width=True)
        else:
            st.caption(f"그림 없음: {api_path} ({r.status_code})")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"그림 로드 실패: {exc}")


def _unavailable(payload: dict, what: str):
    st.warning(f"{what}: 아직 없음 — {payload.get('reason') or payload.get('detail')}")


def _fmt_pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


tabs = st.tabs(
    ["1. 포트폴리오 현황", "2. 강화학습 성과", "3. SHAP 해석", "4. 에이전트 리서치",
     "5. ANOVA 검증 결과", "6. 리스크 모니터링", "운영"]
)

# ------------------------------------------------------------------ 1. 포트폴리오
with tabs[0]:
    st.subheader("학습된 정책의 현재 비중 (POST /optimize, method=drl)")
    method = st.radio("방법", ["drl", "mvo", "equal"], horizontal=True,
                      help="drl = PPO 시드 앙상블. mvo/equal은 비교용")
    if st.button("비중 계산", key="opt"):
        try:
            code, data = _post("/optimize", {"method": method})
            if code != 200:
                detail = data.get("detail", data)
                st.error(f"{code}: {detail.get('error') if isinstance(detail, dict) else detail}")
                if isinstance(detail, dict) and detail.get("hint"):
                    st.caption(detail["hint"])
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("기준일 (as_of)", data.get("as_of") or "—")
                c2.metric("데이터", data.get("data_source") or "—")
                c3.metric("stub", str(data.get("stub")))
                w = pd.Series(data["weights"], name="weight").sort_values(ascending=False)
                st.bar_chart(w)
                if data.get("per_seed"):
                    st.markdown("시드별 비중")
                    st.dataframe(pd.DataFrame(data["per_seed"]).round(4))
                st.caption(data.get("notes", ""))
                if data.get("model"):
                    with st.expander("모델 정보"):
                        st.json(data["model"])
        except Exception as exc:  # noqa: BLE001
            st.error(f"API 호출 실패: {exc}")

    st.subheader("서빙 모델의 Walk-Forward 테스트 1년 (GET /portfolio/history)")
    try:
        h = _get("/portfolio/history")
        if not h["available"]:
            _unavailable(h, "테스트 구간 이력")
        else:
            d = h["data"]
            cum = pd.DataFrame({"ensemble": d["cumulative"], **d["per_seed_cumulative"]},
                               index=pd.to_datetime(d["dates"]))
            st.line_chart(cum - 1.0)
            st.caption(d["note"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"API 호출 실패: {exc}")

# ------------------------------------------------------------------ 2. 성과
with tabs[1]:
    st.subheader("학습 곡선 · lambda 곡선 (GET /training/curves)")
    try:
        tc = _get("/training/curves")
        if not tc["available"]:
            _unavailable(tc, "학습 곡선/lambda 곡선")
        else:
            data = tc["data"]
            lc = data.get("learning")
            if lc:
                pts = pd.DataFrame(lc["points"])
                for rt, g in pts.groupby("reward_type"):
                    st.markdown(f"**reward = {rt}** (에피소드 보상 이동평균)")
                    piv = g.pivot_table(index="timesteps", columns="seed", values="reward_rolling")
                    st.line_chart(piv)
                st.dataframe(pd.DataFrame(lc["convergence"]).round(4))
            if data.get("lambda_sweep"):
                st.markdown("**변형3 lambda 스윕** (테스트 누적수익률·MDD)")
                st.dataframe(pd.DataFrame(data["lambda_sweep"]).round(4))
            for k, p in (data.get("plots") or {}).items():
                _image(p, k)
    except Exception as exc:  # noqa: BLE001
        st.error(f"API 호출 실패: {exc}")

    st.subheader("Walk-Forward 12지표 (GET /backtest)")
    try:
        bt = _get("/backtest")
        if bt.get("source") != "walk_forward":
            st.warning(f"walk-forward 결과 없음 — source={bt.get('source')}")
        else:
            st.info(bt.get("notes", ""))
            st.caption(f"KOSPI: {bt.get('kospi_status')}")
            t = bt.get("targets") or {}
            c1, c2, c3 = st.columns(3)
            c1.metric("DRL 실행 수", t.get("n_drl_runs"))
            c2.metric("샤프 ≥ 동일가중+0.2", f"{t.get('meets_sharpe')} / {t.get('n_drl_runs')}")
            c3.metric("누적 ≥ SPY+10%p", f"{t.get('meets_cumret')} / {t.get('n_drl_runs')}")
            win = pd.DataFrame(bt["windows"])
            st.dataframe(win.round(4), use_container_width=True)
            rows = pd.DataFrame(bt["rows"])
            strat = st.multiselect("전략", sorted(rows.strategy.unique()),
                                   default=sorted(rows.strategy.unique()))
            st.dataframe(rows[rows.strategy.isin(strat)].round(4), use_container_width=True)
            _image("/artifacts/file?root=results&path=figures/wf_cumret_by_window.png",
                   "윈도우별 테스트 누적수익률 (x = Safe-Guard로 절단된 DRL)")
            _image("/artifacts/file?root=results&path=figures/wf_sharpe_by_window.png", "윈도우별 샤프")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API 호출 실패: {exc}")

# ------------------------------------------------------------------ 3. SHAP
with tabs[2]:
    st.subheader("본학습 모델 SHAP (POST /explain)")
    decision = st.selectbox("결정 시점", ["last_decision", "worst_day", "safe_guard"])
    top_k = st.slider("top_k", 3, 20, 10)
    if st.button("설명 조회", key="exp"):
        try:
            code, data = _post("/explain", {"decision": decision, "top_k": int(top_k)})
            st.write(f"mode=`{data.get('mode')}` · stub={data.get('stub')} · latency_ms={data.get('latency_ms')}")
            st.info(data.get("summary", ""))
            if data.get("top_features"):
                df = pd.DataFrame(data["top_features"]).set_index("feature")
                st.bar_chart(df["contribution"])
                if data.get("decisions") and decision not in data["decisions"]:
                    st.caption(f"'{decision}' 결정은 없어 {data['decisions'][0]}을 보여줍니다.")
            for name, path in (data.get("plots") or {}).items():
                if name == "summary" or name == f"force_{decision}":
                    _image(path, name)
            if data.get("global_importance"):
                st.markdown("Summary 대상 자산의 평균 |SHAP|")
                st.dataframe(pd.DataFrame(data["global_importance"]).round(5))
        except Exception as exc:  # noqa: BLE001
            st.error(f"API 호출 실패: {exc}")

# ------------------------------------------------------------------ 4. 리서치
with tabs[3]:
    st.subheader("에이전트 리서치 (POST /research)")
    st.error(
        "STUB — 뉴스 코퍼스가 수집되지 않았습니다. 아래 결과는 LangGraph 형태의 흐름과 "
        "출처 필드를 보여주는 데모이며 실제 뉴스 분석이 아닙니다. 리스크 태그도 강화학습 "
        "학습에 사용되지 않았습니다(portfolio_risk = 0)."
    )
    query = st.text_input("질문", "market risk outlook")
    tickers_r = st.text_input("티커", "SPY,QQQ,TLT")
    if st.button("리서치 실행", key="res"):
        try:
            code, data = _post("/research", {
                "query": query, "tickers": [t.strip() for t in tickers_r.split(",") if t.strip()],
                "n_events_per_ticker": 2, "top_k": 5})
            st.write(f"stub={data.get('stub')} · verify_ok={data.get('verify_ok')}")
            st.markdown("**추론 과정 (현재 생각 로그)**")
            for node, lat in zip(data.get("node_trace") or [], data.get("node_latencies_ms") or []):
                st.code(f"[{node}] {lat.get('latency_ms')} ms")
            for step in data.get("plan") or []:
                st.write(f"- {step}")
            st.markdown("**결과**")
            st.write(data.get("report_excerpt", ""))
            if data.get("citations"):
                st.markdown("**출처 (스텁 코퍼스)**")
                st.dataframe(pd.DataFrame(data["citations"]))
            if data.get("risk_tags"):
                st.markdown("**리스크 태그 (계약 형식)**")
                st.dataframe(pd.DataFrame(data["risk_tags"]))
        except Exception as exc:  # noqa: BLE001
            st.error(f"API 호출 실패: {exc}")

# ------------------------------------------------------------------ 5. ANOVA
with tabs[4]:
    st.subheader("ANOVA 검증 (GET /anova/results — Walk-Forward 결과 CSV로 계산)")
    try:
        a = _get("/anova/results")
        if not a["available"]:
            _unavailable(a, "ANOVA 결과")
        else:
            d = a["data"]
            rows = []
            for key, label in (
                ("validation1_reward_oneway", "검증1 보상 3종 (누적수익률)"),
                ("validation1_reward_oneway_sharpe", "검증1 보상 3종 (샤프)"),
                ("validation1_excluding_truncated", "검증1 (절단 행 제외, 보조)"),
                ("validation2_strategy_oneway", "검증2 DRL vs MVO vs 동일가중"),
                ("validation2_excluding_truncated_windows", "검증2 (절단 윈도우 제외, 보조)"),
            ):
                v = d.get(key) or {}
                rows.append({"검증": label, "F": v.get("f_stat"), "p": v.get("p_value"),
                             "eta²": v.get("eta_squared"), "Tukey": "수행" if v.get("tukey_records") else
                             ("불필요 (p≥0.05)" if v else "—")})
            st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)
            blk = d.get("validation1_reward_blocked_by_window")
            if blk:
                st.markdown("**보조: 보상 × 윈도우 (윈도우 블록 통제)**")
                st.dataframe(pd.DataFrame(blk["table"]).round(4))
                if blk.get("tukey_records_window_centered"):
                    st.caption(blk.get("tukey_note", ""))
                    st.dataframe(pd.DataFrame(blk["tukey_records_window_centered"]))
            v3 = d.get("validation3_strategy_x_regime") or {}
            st.markdown("**검증3 전략 × 시장국면 (Two-way)**")
            if v3.get("status") == "computed":
                st.dataframe(pd.DataFrame(v3["table"]).round(4))
            else:
                st.warning(v3.get("reason", ""))
                st.json(v3.get("reported_in_pr7", {}))
            st.caption(f"KOSPI: {d.get('kospi')}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API 호출 실패: {exc}")

# ------------------------------------------------------------------ 6. 리스크
with tabs[5]:
    st.subheader("리스크 모니터링")
    try:
        h = _get("/portfolio/history")
        if not h["available"]:
            _unavailable(h, "서빙 모델 테스트 이력")
        else:
            d = h["data"]
            c1, c2, c3 = st.columns(3)
            c1.metric("VaR 95% (일)", _fmt_pct(d["var_95"]))
            c2.metric("CVaR 95% (일)", _fmt_pct(d["cvar_95"]))
            c3.metric("MDD", _fmt_pct(d["mdd"]), help="Safe-Guard 한도 15%")
            st.progress(min(1.0, d["mdd"] / d["mdd_limit"]), text=f"MDD / Safe-Guard 한도 = {d['mdd'] / d['mdd_limit']:.0%}")
            dd = pd.DataFrame({"drawdown": d["drawdown"], "limit": [d["mdd_limit"]] * len(d["dates"])},
                              index=pd.to_datetime(d["dates"]))
            st.line_chart(dd)
            if d["safe_guard_events"]:
                for e in d["safe_guard_events"]:
                    st.error(f"Safe-Guard 발동: seed {e['seed']} — {e['date']} (이후 현금 보유)")
            else:
                st.success("서빙 모델 테스트 구간에서 Safe-Guard 발동 없음")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API 호출 실패: {exc}")
    st.markdown("**Walk-Forward 백테스트에서 Safe-Guard로 절단된 DRL 실행**")
    try:
        bt = _get("/backtest")
        tr = pd.DataFrame(bt.get("truncated_rows") or [])
        if tr.empty:
            st.success("절단된 실행 없음")
        else:
            st.dataframe(tr.round(4), use_container_width=True)
            st.caption("n_days = 실제 평가된 거래일 (CAGR 역산). 이 결과는 수정 전 코드로 만든 것이며 재실행 필요.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API 호출 실패: {exc}")

# ------------------------------------------------------------------ 운영
with tabs[6]:
    st.subheader("GET /health")
    if st.button("health", key="hlt"):
        try:
            h = _get("/health")
            st.success(f"API v{h.get('version')} · educational={h.get('educational')}")
            st.json(h.get("models", {}))
            st.write("endpoints: " + ", ".join(f"`{p}`" for p in h.get("endpoints", [])))
        except Exception as exc:  # noqa: BLE001
            st.error(f"API 호출 실패: {exc}")
    st.subheader("리스크 태그 → PortfolioEnv 배선 데모 (POST /risk-tags/apply, 스텁)")
    if st.button("배선 데모", key="wire"):
        try:
            code, data = _post("/risk-tags/apply", {"tickers": ["SPY", "QQQ", "TLT"], "source": "mock",
                                                    "run_env_demo": True, "env_steps": 8, "seed": 0})
            st.json({k: data.get(k) for k in ("source", "stub", "panel_summary", "env_demo", "notes")})
        except Exception as exc:  # noqa: BLE001
            st.error(f"API 호출 실패: {exc}")

st.divider()
st.caption(DISCLAIMER)
