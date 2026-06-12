"""Węzły grafu agenta.

Każdy węzeł to funkcja: (state) -> częściowy stan do scalenia.
Zwracamy TYLKO pola, które dany węzeł produkuje (patrz state.py).
"""

from __future__ import annotations

import json
from collections.abc import Callable

from app.agent.state import AgentState, Report
from app.llm.client import LLMClient
from app.rag.embeddings import Embedder
from app.rag.store import VectorStore
from app.tools.market_data import MarketDataError, get_market_data
from app.tools.metrics import calc_metrics
from app.tools.news import get_news
from app.tools.rag import rag_search


def router_node(state: AgentState) -> AgentState:
    """ROUTER — ustala plan: które narzędzia uruchomić.

    market_data, metrics i news bierzemy zawsze (to podstawa analizy spółki).
    O RAG decydujemy osobno, bo jest WARUNKOWY — patrz should_use_rag().
    """
    return {"plan": ["market_data", "metrics", "news", "rag?"]}


def market_data_node(state: AgentState) -> AgentState:
    """Pobiera dane rynkowe dla tickera ze stanu."""
    try:
        return {"market_data": get_market_data(state["ticker"])}
    except MarketDataError as exc:
        return {"error": str(exc)}


def metrics_node(state: AgentState) -> AgentState:
    """Liczy metryki z danych rynkowych ze stanu."""
    if "market_data" not in state:
        return {"error": "Brak danych rynkowych do policzenia metryk."}
    return {"metrics": calc_metrics(state["market_data"])}


def news_node(state: AgentState) -> AgentState:
    """Pobiera ostatnie newsy spółki."""
    return {"news": get_news(state["ticker"])}


def make_rag_node(store: VectorStore, embedder: Embedder) -> Callable[[AgentState], AgentState]:
    """Węzeł rag z wstrzykniętym store + embedder (DI, jak przy LLM)."""

    def rag_node(state: AgentState) -> AgentState:
        ticker = state["ticker"]
        # Stałe zapytanie analityczne — szukamy kontekstu o perspektywach/ryzykach.
        query = f"{ticker} perspektywy wzrostu ryzyka wyniki"
        return {"rag_context": rag_search(ticker, query, store, embedder)}

    return rag_node


def make_should_use_rag(store: VectorStore) -> Callable[[AgentState], str]:
    """Funkcja decyzyjna conditional edge: włączyć RAG czy iść prosto do syntezy.

    To REALNA decyzja agenta oparta o stan świata: jeśli w bazie jest kontekst
    dla tego tickera -> 'rag', w przeciwnym razie -> 'synthesize'.
    """

    def should_use_rag(state: AgentState) -> str:
        return "rag" if store.has_documents(state["ticker"]) else "synthesize"

    return should_use_rag


def make_synthesize_node(llm: LLMClient) -> Callable[[AgentState], AgentState]:
    """Tworzy węzeł synthesize z WSTRZYKNIĘTYM klientem LLM (dependency injection)."""

    def synthesize_node(state: AgentState) -> AgentState:
        if "error" in state:
            return {}  # nie syntetyzujemy, gdy wcześniej był błąd

        metrics = state["metrics"]
        data = state["market_data"]
        news = state.get("news", [])
        rag_context = state.get("rag_context", [])

        # Do modelu podajemy POLICZONE liczby + kontekst (newsy, RAG). Prosimy go
        # TYLKO o ocenę jakościową, nie o liczby (anty-halucynacja).
        system = (
            "Jesteś analitykiem akcji. Na podstawie metryk ORAZ kontekstu (newsy, "
            "fragmenty z bazy wiedzy) wydaj zwięzłą, ostrożną ocenę. Zwróć WYŁĄCZNIE "
            "JSON o kluczach: recommendation (Pozytywna/Neutralna/Negatywna), "
            "rationale (1-3 zdania), risks (lista 2-4 krótkich ryzyk). Nie podawaj "
            "liczb, których nie ma w danych wejściowych."
        )
        user = json.dumps(
            {
                "ticker": data.ticker,
                "current_price": data.current_price,
                "currency": data.currency,
                "momentum_12_1": metrics.momentum_12_1,
                "upside_vs_consensus": metrics.upside,
                "news_titles": [n.title for n in news],
                "rag_context": rag_context,
            },
            ensure_ascii=False,
        )

        raw = llm.complete_json(system, user)

        report = Report(
            recommendation=str(raw.get("recommendation", "Neutralna")),
            rationale=str(raw.get("rationale", "")),
            risks=[str(r) for r in raw.get("risks", [])],
            metrics=metrics,                       # NASZE liczby, nie z LLM
            news_used=[n.title for n in news],     # transparentność źródeł
        )
        return {"report": report}

    return synthesize_node
