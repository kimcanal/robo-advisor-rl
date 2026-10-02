"""Streamlit dashboard — API client only (no local model load)."""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Robo-Advisor Dashboard", layout="wide")
st.title("라스트댄스 — Streamlit (API client)")
st.caption(
    "Educational demo. Not investment advice. Backtests do not guarantee future returns. "
    f"API: `{API_BASE}`"
)

tabs = st.tabs(
    [
        "1. Overview",
        "2. Optimize",
        "3. Explain",
        "4. Research",
        "5. Risk-tag wiring",
        "6. Backtest",
        "7. ANOVA",
        "8. Health / Settings",
    ]
)


def _get(path: str, **params):
    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{API_BASE}{path}", params=params or None)
        r.raise_for_status()
        return r.json()


def _post(path: str, payload: dict):
    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{API_BASE}{path}", json=payload)
        r.raise_for_status()
        return r.json()


with tabs[0]:
    st.subheader("Overview")
    st.write(
        "Architecture: **Streamlit → HTTP → FastAPI → rl.* / rag stub** "
        "(plan → retrieve → tag_risk → verify → summarize)."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Ping /health", key="ov_hlt"):
            try:
                h = _get("/health")
                st.success(h)
            except Exception as exc:
                st.error(f"API call failed: {exc}")
    with col2:
        st.markdown("**Env contract**")
        st.code("{ticker, risk_score[0,1], tag, ts} → PortfolioEnv.portfolio_risk")
    with col3:
        st.markdown("**Disclaimer**")
        st.warning("Synthetic / educational only — not investment advice.")

with tabs[1]:
    st.subheader("Optimize")
    method = st.selectbox("method", ["mvo", "equal"])
    tickers_txt = st.text_input("tickers (comma)", "SPY,QQQ,AGG")
    if st.button("Run /optimize", key="opt"):
        try:
            tickers = [t.strip() for t in tickers_txt.split(",") if t.strip()]
            data = _post(
                "/optimize",
                {"method": method, "lookback_days": 252, "tickers": tickers},
            )
            weights = data.get("weights") or {}
            st.metric("method", data.get("method", method))
            if weights:
                st.bar_chart(pd.Series(weights, name="weight"))
                st.dataframe(pd.DataFrame([weights]).T.rename(columns={0: "weight"}))
            if data.get("notes"):
                st.caption(data["notes"])
            with st.expander("raw JSON"):
                st.json(data)
        except Exception as exc:
            st.error(f"API call failed: {exc}")

with tabs[2]:
    st.subheader("Explain (SHAP path or stub)")
    asset_index = st.number_input("asset_index", min_value=0, value=0, step=1)
    top_k = st.slider("top_k", 1, 15, 5)
    if st.button("Run /explain", key="exp"):
        try:
            data = _post(
                "/explain",
                {
                    "tickers": ["SPY", "QQQ", "AGG"],
                    "asset_index": int(asset_index),
                    "top_k": int(top_k),
                },
            )
            st.write(f"**mode:** `{data.get('mode')}` · stub={data.get('stub')} · "
                     f"latency_ms={data.get('latency_ms')}")
            st.info(data.get("summary", ""))
            feats = data.get("top_features") or []
            if feats:
                df = pd.DataFrame(feats)
                st.bar_chart(df.set_index("feature")["contribution"])
                st.dataframe(df)
            with st.expander("raw JSON"):
                st.json(data)
        except Exception as exc:
            st.error(f"API call failed: {exc}")

with tabs[3]:
    st.subheader("Research (RAG plan / exec / verify)")
    query = st.text_input("query", "market risk outlook")
    tickers_r = st.text_input("tickers", "SPY,QQQ,TLT", key="res_tickers")
    if st.button("Run /research", key="res"):
        try:
            tickers = [t.strip() for t in tickers_r.split(",") if t.strip()]
            data = _post(
                "/research",
                {"query": query, "tickers": tickers, "n_events_per_ticker": 2, "top_k": 5},
            )
            st.write(
                f"**verify_ok:** {data.get('verify_ok')} · "
                f"trace: `{' → '.join(data.get('node_trace') or [])}` · "
                f"latency_ms={data.get('latency_ms')}"
            )
            st.markdown("**Plan**")
            for step in data.get("plan") or []:
                st.write(f"- {step}")
            st.markdown("**Report excerpt**")
            st.write(data.get("report_excerpt", ""))
            tags = data.get("risk_tags") or []
            if tags:
                st.markdown("**Risk tags (env contract)**")
                st.dataframe(pd.DataFrame(tags))
            cites = data.get("citations") or []
            if cites:
                st.markdown("**Citations (placeholders)**")
                st.dataframe(pd.DataFrame(cites))
            if data.get("verify_notes"):
                st.caption("verify: " + " | ".join(data["verify_notes"]))
            st.caption(data.get("env_contract", ""))
            with st.expander("raw JSON"):
                st.json(data)
        except Exception as exc:
            st.error(f"API call failed: {exc}")


with tabs[4]:
    st.subheader("Risk-tag wiring (panel → PortfolioEnv.portfolio_risk)")
    st.caption(
        "Educational stub: POST /risk-tags/apply builds a causal risk_score_panel "
        "and optionally steps a short PortfolioEnv on dummy data. Not investment advice."
    )
    tickers_w = st.text_input("tickers", "SPY,QQQ,TLT", key="wire_tickers")
    source = st.selectbox("source (when no client tags)", ["mock", "rag_graph"], key="wire_src")
    run_env = st.checkbox("run_env_demo", value=True, key="wire_env")
    env_steps = st.slider("env_steps", 1, 20, 8, key="wire_steps")
    if st.button("Run /risk-tags/apply", key="wire"):
        try:
            tickers = [t.strip() for t in tickers_w.split(",") if t.strip()]
            data = _post(
                "/risk-tags/apply",
                {
                    "tickers": tickers,
                    "source": source,
                    "run_env_demo": run_env,
                    "env_steps": int(env_steps),
                    "n_events_per_ticker": 3,
                    "seed": 0,
                },
            )
            st.write(
                f"**source:** `{data.get('source')}` · stub={data.get('stub')} · "
                f"latency_ms={data.get('latency_ms')}"
            )
            st.caption(data.get("env_contract", ""))
            panel = data.get("panel_summary") or {}
            if panel:
                st.markdown("**Panel summary (causal)**")
                st.json(
                    {
                        k: panel[k]
                        for k in (
                            "n_dates",
                            "n_tickers",
                            "nonzero_cells",
                            "global_mean",
                            "global_max",
                            "date_start",
                            "date_end",
                        )
                        if k in panel
                    }
                )
            tags = data.get("risk_tags") or []
            if tags:
                st.markdown("**Risk tags**")
                st.dataframe(pd.DataFrame(tags))
            env_demo = data.get("env_demo") or {}
            if env_demo:
                st.markdown("**Env demo (portfolio_risk samples)**")
                st.write(
                    f"obs_dim={env_demo.get('obs_dim')} · "
                    f"saw_nonzero_risk={env_demo.get('saw_nonzero_risk')} · "
                    f"mean={env_demo.get('portfolio_risk_mean')} · "
                    f"max={env_demo.get('portfolio_risk_max')}"
                )
                samples = env_demo.get("sample_portfolio_risk") or []
                if samples:
                    st.dataframe(pd.DataFrame(samples))
            if data.get("notes"):
                st.caption(data["notes"])
            with st.expander("raw JSON"):
                st.json(data)
        except Exception as exc:
            st.error(f"API call failed: {exc}")

with tabs[5]:
    st.subheader("Backtest metrics (+ synth SPY/KOSPI)")
    n_days = st.slider("n_days", 60, 504, 252)
    include_benchmark = st.checkbox("include_benchmark", value=True)
    if st.button("Run /backtest", key="bt"):
        try:
            data = _get(
                "/backtest",
                n_days=n_days,
                seed=0,
                include_benchmark=include_benchmark,
                benchmarks="spy,kospi",
            )
            st.write(
                f"**method:** `{data.get('method')}` · latency_ms={data.get('latency_ms')}"
            )
            metrics = data.get("metrics") or {}
            if metrics:
                cols = st.columns(min(4, len(metrics)))
                for i, key in enumerate(["sharpe", "mdd", "cagr", "ann_vol"]):
                    if key in metrics and metrics[key] is not None:
                        cols[i % 4].metric(key, f"{metrics[key]:.4f}")
                st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}))
            bench = data.get("benchmark_metrics") or {}
            if bench:
                st.markdown("**Benchmark metrics (synthetic)**")
                rows = []
                for name, m in bench.items():
                    row = {"benchmark": name}
                    row.update(m)
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows))
            st.caption(data.get("notes", ""))
            st.caption(data.get("latency_notes", ""))
            with st.expander("raw JSON"):
                st.json(data)
        except Exception as exc:
            st.error(f"API call failed: {exc}")

with tabs[6]:
    st.subheader("ANOVA (합성 시리즈 · 교육용)")
    st.caption("합성 수익률에 대한 통계 데모입니다. 투자 자문이 아닙니다.")
    mode = st.selectbox("mode", ["one_way", "two_way"])
    n_obs = st.slider("n_obs", 30, 500, 120)
    if st.button("Run /anova", key="anova"):
        try:
            data = _post("/anova", {"mode": mode, "n_obs": n_obs, "seed": 0})
            st.write(f"**groups:** {', '.join(data.get('groups') or [])}")
            st.json(data.get("result") or {})
            st.caption(data.get("notes", ""))
            with st.expander("raw JSON"):
                st.json(data)
        except Exception as exc:
            st.error(f"API call failed: {exc}")

with tabs[7]:
    st.subheader("Health / Settings")
    st.write(f"`API_BASE_URL` = `{API_BASE}`")
    if st.button("GET /health", key="hlt"):
        try:
            st.success(_get("/health"))
        except Exception as exc:
            st.error(f"API call failed: {exc}")
