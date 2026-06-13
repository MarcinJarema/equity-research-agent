"""Streamlit UI — wizualny frontend nad API agenta.

Architektura: UI (tu) -> HTTP POST /analyze -> agent LangGraph -> raport.
UI jest CIENKIM klientem usługi — nie zna logiki agenta, tylko woła API. To ta
sama separacja, którą pokazujesz na rozmowie: serwis + interfejs nad nim.

Uruchomienie:
    1) API:  uvicorn app.main:app          (musi działać, domyślnie :8000)
    2) UI:   streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("ERA_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Equity Research Agent", page_icon="📈", layout="centered")


# --- Pasek boczny: konfiguracja + status API ---
with st.sidebar:
    st.header("⚙️ Ustawienia")
    api_url = st.text_input("Adres API", value=API_URL, help="Gdzie działa FastAPI agenta.")
    try:
        health = httpx.get(f"{api_url}/health", timeout=3).json()
        st.success(f"API online · trace: {health.get('tracing')}")
    except Exception:
        st.error("API offline — uruchom:\n`uvicorn app.main:app`")
    st.divider()
    st.caption(
        "UI → POST /analyze → agent LangGraph "
        "(router → market_data → metrics → news → RAG → synteza LLM)."
    )


# --- Nagłówek ---
st.title("📈 Equity Research Agent")
st.caption("Agentowy raport spółki: dane rynkowe → metryki → newsy → RAG → synteza LLM.")

col_in, col_btn = st.columns([3, 1])
ticker = col_in.text_input(
    "Ticker spółki", value="AAPL", max_chars=10, label_visibility="collapsed",
    placeholder="np. AAPL, MSFT, NVDA",
).strip().upper()
analyze = col_btn.button("Analizuj", type="primary", use_container_width=True)


def _pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


# --- Akcja: wywołanie API i render raportu ---
if analyze and ticker:
    with st.spinner(f"Analizuję {ticker}…"):
        try:
            resp = httpx.post(f"{api_url}/analyze", json={"ticker": ticker}, timeout=90)
        except Exception as exc:
            st.error(f"Nie udało się połączyć z API: {exc}")
            st.stop()

    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text)
        st.error(f"Błąd {resp.status_code}: {detail}")
        st.stop()

    data = resp.json()

    # Rekomendacja jako kolorowy nagłówek.
    rec = data["recommendation"]
    color = {"Pozytywna": "green", "Neutralna": "orange", "Negatywna": "red"}.get(rec, "gray")
    st.markdown(f"## {ticker} — rekomendacja: :{color}[{rec}]")

    # Metryki: momentum i upside (delta koloruje się auto: + zielono, − czerwono).
    m = data.get("metrics", {})
    c1, c2 = st.columns(2)
    c1.metric("Momentum 12-1", _pct(m.get("momentum_12_1")), delta=_pct(m.get("momentum_12_1")))
    c2.metric("Upside vs konsensus", _pct(m.get("upside")), delta=_pct(m.get("upside")))

    st.markdown("### 🧠 Uzasadnienie")
    st.info(data.get("rationale", "—"))

    risks = data.get("risks") or []
    if risks:
        st.markdown("### ⚠️ Ryzyka")
        for r in risks:
            st.markdown(f"- {r}")

    news = data.get("news_used") or []
    if news:
        with st.expander(f"📰 Źródła — newsy ({len(news)})"):
            for n in news:
                st.markdown(f"- {n}")

    with st.expander("🔧 Surowa odpowiedź API (JSON)"):
        st.json(data)

    st.warning(data.get("disclaimer", ""))

elif analyze and not ticker:
    st.warning("Podaj ticker spółki.")
