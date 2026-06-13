"""Streamlit UI — wizualny frontend nad API agenta.

Architektura: UI (tu) -> HTTP (FastAPI) -> agent / regułowy rating -> wynik.
UI jest CIENKIM klientem usługi — nie zna logiki, tylko woła API. To ta sama
separacja, którą pokazujesz na rozmowie: serwis + interfejs nad nim.

Dwie zakładki:
  • Analiza      — pełny raport LLM dla jednej spółki (POST /analyze),
  • Porównanie   — regułowy rating Strong Buy/Buy/Hold/Sell dla wielu spółek (POST /compare).

Uruchomienie:
    1) API:  uvicorn app.main:app
    2) UI:   streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("ERA_API_URL", "http://127.0.0.1:8000")

RATING_ORDER = ["Strong Buy", "Buy", "Hold", "Sell"]
RATING_COLOR = {"Strong Buy": "green", "Buy": "blue", "Hold": "orange", "Sell": "red"}
# Paleta hex dla wykresu (Altair) — w kolejności RATING_ORDER.
RATING_HEX = ["#2e7d32", "#1565c0", "#ef6c00", "#c62828"]

st.set_page_config(page_title="Equity Research Agent", page_icon="📈", layout="wide")


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
        "Analiza: UI → /analyze → agent LangGraph (LLM).\n\n"
        "Porównanie: UI → /compare → regułowy rating (bez LLM)."
    )


@st.cache_data(ttl=300)
def fetch_universe(url: str) -> list[str]:
    """Domyślne uniwersum spółek z API (cache, by nie pytać co rerun)."""
    try:
        return httpx.get(f"{url}/universe", timeout=5).json()
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"]


def _pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _rev_label(negative: bool) -> str:
    return "↓ negatywne" if negative else "–"


st.title("📈 Equity Research Agent")

tab_single, tab_compare = st.tabs(["🔍 Analiza spółki", "📊 Porównanie spółek"])


# =========================== ZAKŁADKA: ANALIZA ===========================
with tab_single:
    st.caption("Pełny raport: dane rynkowe → metryki → newsy → RAG → synteza LLM.")

    single_universe = fetch_universe(api_url)
    col_in, col_btn = st.columns([3, 1])
    picked = col_in.selectbox(
        "Ticker spółki",
        options=single_universe,
        index=single_universe.index("AAPL") if "AAPL" in single_universe else 0,
        accept_new_options=True,   # można też wpisać własny ticker spoza listy
        label_visibility="collapsed",
        placeholder="Wybierz lub wpisz ticker…",
    )
    ticker = (picked or "").strip().upper()
    analyze = col_btn.button("Analizuj", type="primary", use_container_width=True)

    if analyze and ticker:
        with st.spinner(f"Analizuję {ticker}…"):
            try:
                resp = httpx.post(f"{api_url}/analyze", json={"ticker": ticker}, timeout=90)
            except Exception as exc:
                st.error(f"Nie udało się połączyć z API: {exc}")
                st.stop()

        if resp.status_code != 200:
            st.error(f"Błąd {resp.status_code}: {resp.json().get('detail', resp.text)}")
            st.stop()

        data = resp.json()
        rec = data["recommendation"]
        color = {"Pozytywna": "green", "Neutralna": "orange", "Negatywna": "red"}.get(rec, "gray")
        st.markdown(f"## {ticker} — rekomendacja: :{color}[{rec}]")

        m = data.get("metrics", {})
        c1, c2 = st.columns(2)
        c1.metric("Momentum 12-1", _pct(m.get("momentum_12_1")), delta=_pct(m.get("momentum_12_1")))
        c2.metric("Upside vs konsensus", _pct(m.get("upside")), delta=_pct(m.get("upside")))

        st.markdown("### 🧠 Uzasadnienie")
        st.info(data.get("rationale", "—"))

        if data.get("risks"):
            st.markdown("### ⚠️ Ryzyka")
            for r in data["risks"]:
                st.markdown(f"- {r}")

        if data.get("news_used"):
            with st.expander(f"📰 Źródła — newsy ({len(data['news_used'])})"):
                for n in data["news_used"]:
                    st.markdown(f"- {n}")

        with st.expander("🔧 Surowa odpowiedź API (JSON)"):
            st.json(data)

        st.warning(data.get("disclaimer", ""))

    elif analyze and not ticker:
        st.warning("Podaj ticker spółki.")


# ========================== ZAKŁADKA: PORÓWNANIE ==========================
with tab_compare:
    st.caption(
        "Regułowy rating dla wielu spółek naraz: momentum + upside + rewizje analityków. "
        "**Negatywne rewizje od 2 miesięcy degradują ocenę.** Bez LLM — szybko i tanio."
    )

    universe = fetch_universe(api_url)
    chosen = st.multiselect(
        "Spółki do porównania", options=universe, default=universe,
        help="Domyślnie zaznaczone wszystkie. Odznacz, by zawęzić.",
    )
    extra = st.text_input(
        "Dodatkowe tickery (po przecinku)", value="", placeholder="np. INTC, NFLX",
    )
    compare = st.button("Porównaj", type="primary")

    if compare:
        tickers = list(chosen)
        tickers += [t.strip().upper() for t in extra.split(",") if t.strip()]
        if not tickers:
            st.warning("Wybierz przynajmniej jedną spółkę.")
            st.stop()

        spinner_msg = f"Liczę rating dla {len(tickers)} spółek… (pobieranie danych może potrwać)"
        with st.spinner(spinner_msg):
            try:
                resp = httpx.post(f"{api_url}/compare", json={"tickers": tickers}, timeout=180)
            except Exception as exc:
                st.error(f"Nie udało się połączyć z API: {exc}")
                st.stop()

        if resp.status_code != 200:
            st.error(f"Błąd {resp.status_code}: {resp.json().get('detail', resp.text)}")
            st.stop()

        rows = resp.json()
        ok_rows = [r for r in rows if r.get("rating")]
        err_rows = [r for r in rows if r.get("error")]

        # Pogrupowane, kolorowe kubełki — to jest sedno: które spółki w którym koszyku.
        st.markdown("### Koszyki ratingowe")
        cols = st.columns(4)
        for col, rating in zip(cols, RATING_ORDER, strict=True):
            bucket = [r["ticker"] for r in ok_rows if r["rating"] == rating]
            with col:
                st.markdown(f"#### :{RATING_COLOR[rating]}[{rating}]")
                st.markdown(f"**{len(bucket)}**")
                st.markdown("\n".join(f"- {t}" for t in bucket) if bucket else "_brak_")

        if ok_rows:
            df = pd.DataFrame(ok_rows)
            df["_order"] = df["rating"].map({r: i for i, r in enumerate(RATING_ORDER)})
            df = df.sort_values(["_order", "upside"], ascending=[True, False])

            # Słupki porównawcze, kolorowane wg ratingu. Przełącznik metryki.
            st.markdown("### Wykres porównawczy")
            metric_label = st.radio(
                "Metryka",
                ["Upside", "Momentum 12-1"],
                horizontal=True,
                label_visibility="collapsed",
            )
            field = "upside" if metric_label == "Upside" else "momentum_12_1"
            chart_df = df[["ticker", "rating"]].copy()
            chart_df["wartość"] = pd.to_numeric(df[field], errors="coerce") * 100  # w %
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("ticker:N", sort="-y", title="Spółka"),
                    y=alt.Y("wartość:Q", title=f"{metric_label} [%]"),
                    color=alt.Color(
                        "rating:N",
                        scale=alt.Scale(domain=RATING_ORDER, range=RATING_HEX),
                        title="Rating",
                    ),
                    tooltip=[
                        alt.Tooltip("ticker:N", title="Spółka"),
                        alt.Tooltip("rating:N", title="Rating"),
                        alt.Tooltip("wartość:Q", title=metric_label, format=".1f"),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)

            # Tabela szczegółowa, posortowana wg ratingu (najlepsze u góry).
            st.markdown("### Szczegóły")
            view = pd.DataFrame({
                "Ticker": df["ticker"],
                "Rating": df["rating"],
                "Momentum 12-1": df["momentum_12_1"].map(_pct),
                "Upside": df["upside"].map(_pct),
                "Rewizje 2m": df["revisions_negative_2m"].map(_rev_label),
            })
            st.dataframe(view, use_container_width=True, hide_index=True)

        if err_rows:
            with st.expander(f"⚠️ Spółki z błędem ({len(err_rows)})"):
                for r in err_rows:
                    st.markdown(f"- **{r['ticker']}**: {r['error']}")

        st.warning(
            "Rating jest regułowy i edukacyjny — NIE porada inwestycyjna. "
            "Nie podejmuj decyzji inwestycyjnych na tej podstawie."
        )
