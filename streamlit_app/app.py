"""Minimal Streamlit dashboard — API client only (no local model load)."""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Robo-Advisor Dashboard (stub)", layout="wide")
st.title("라스트댄스 — Streamlit stub (API only)")
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
        "5. Backtest",
        "6. ANOVA",
        "7. Health / Settings",
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
    st.subheader("Overview (placeholder)")
    st.write(
        "Shell that only talks to FastAPI. "
        "Wire real charts once backend endpoints return production payloads."
    )
    st.info("Architecture: Streamlit → HTTP → FastAPI → rl.* / rag stub")

with tabs[1]:
    st.subheader("Optimize")
    method = st.selectbox("method", ["mvo", "equal"])
    if st.button("Run /optimize", key="opt"):
        try:
            data = _post("/optimize", {"method": method, "lookback_days": 252})
            st.json(data)
        except Exception as exc:
            st.error(f"API call failed: {exc}")

with tabs[2]:
    st.subheader("Explain (SHAP stub)")
    if st.button("Run /explain", key="exp"):
        try:
            data = _post("/explain", {"tickers": ["SPY", "QQQ", "AGG"], "asset_index": 0})
            st.json(data)
        except Exception as exc:
            st.error(f"API call failed: {exc}")

with tabs[3]:
    st.subheader("Research (RAG stub)")
    query = st.text_input("query", "market risk outlook")
    if st.button("Run /research", key="res"):
        try:
            data = _post("/research", {"query": query, "tickers": ["SPY", "QQQ", "TLT"]})
            st.json(data)
        except Exception as exc:
            st.error(f"API call failed: {exc}")

with tabs[4]:
    st.subheader("Backtest metrics")
    n_days = st.slider("n_days", 60, 504, 252)
    if st.button("Run /backtest", key="bt"):
        try:
            data = _get("/backtest", n_days=n_days, seed=0)
            st.json(data)
        except Exception as exc:
            st.error(f"API call failed: {exc}")

with tabs[5]:
    st.subheader("ANOVA (합성 시리즈 · 교육용)")
    st.caption("합성 수익률에 대한 통계 데모입니다. 투자 자문이 아닙니다.")
    mode = st.selectbox("mode", ["one_way", "two_way"])
    n_obs = st.slider("n_obs", 30, 500, 120)
    if st.button("Run /anova", key="anova"):
        try:
            data = _post("/anova", {"mode": mode, "n_obs": n_obs, "seed": 0})
            st.json(data)
        except Exception as exc:
            st.error(f"API call failed: {exc}")

with tabs[6]:
    st.subheader("Health / Settings")
    st.write(f"`API_BASE_URL` = `{API_BASE}`")
    if st.button("GET /health", key="hlt"):
        try:
            st.success(_get("/health"))
        except Exception as exc:
            st.error(f"API call failed: {exc}")
