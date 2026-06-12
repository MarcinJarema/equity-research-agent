"""Węzły grafu agenta.

Każdy węzeł to funkcja: (state) -> częściowy stan do scalenia.
Zwracamy TYLKO pola, które dany węzeł produkuje (patrz state.py).
"""

from __future__ import annotations

import json
from collections.abc import Callable

from app.agent.state import AgentState, Report
from app.llm.client import LLMClient
from app.tools.market_data import MarketDataError, get_market_data
from app.tools.metrics import calc_metrics


def router_node(state: AgentState) -> AgentState:
    """ROUTER — decyduje, które narzędzia uruchomić.

    DLACZEGO to osobny węzeł, skoro w MVP zawsze bierzemy te same dane:
    bo to miejsce decyzji. W ETAPIE 3 dołożymy 'news' i 'rag_search', które są
    OPCJONALNE — wtedy router (regułowo lub przez LLM) wybierze podzbiór narzędzi.
    Dziś plan jest stały: dane rynkowe są podstawą każdej analizy spółki.
    """
    return {"plan": ["market_data", "metrics"]}


def market_data_node(state: AgentState) -> AgentState:
    """Pobiera dane rynkowe dla tickera ze stanu."""
    try:
        data = get_market_data(state["ticker"])
        return {"market_data": data}
    except MarketDataError as exc:
        return {"error": str(exc)}


def metrics_node(state: AgentState) -> AgentState:
    """Liczy metryki z danych rynkowych ze stanu."""
    if "market_data" not in state:
        return {"error": "Brak danych rynkowych do policzenia metryk."}
    return {"metrics": calc_metrics(state["market_data"])}


def make_synthesize_node(llm: LLMClient) -> Callable[[AgentState], AgentState]:
    """Tworzy węzeł synthesize z WSTRZYKNIĘTYM klientem LLM (dependency injection).

    Zwracamy domknięcie, bo węzeł w LangGraph to funkcja (state) -> state,
    a klient LLM chcemy podać z zewnątrz (prawdziwy w aplikacji, atrapa w testach).
    """

    def synthesize_node(state: AgentState) -> AgentState:
        if "error" in state:
            return {}  # nie syntetyzujemy, gdy wcześniej był błąd

        metrics = state["metrics"]
        data = state["market_data"]

        # Do modelu podajemy POLICZONE liczby. Prosimy go TYLKO o ocenę jakościową,
        # nie o liczby — liczby dokładamy sami niżej (anty-halucynacja).
        system = (
            "Jesteś analitykiem akcji. Na podstawie podanych metryk wydaj zwięzłą, "
            "ostrożną ocenę. Zwróć WYŁĄCZNIE JSON o kluczach: "
            "recommendation (Pozytywna/Neutralna/Negatywna), rationale (1-3 zdania), "
            "risks (lista 2-4 krótkich ryzyk). Nie podawaj żadnych liczb, których "
            "nie ma w danych wejściowych."
        )
        user = json.dumps(
            {
                "ticker": data.ticker,
                "current_price": data.current_price,
                "currency": data.currency,
                "momentum_12_1": metrics.momentum_12_1,
                "upside_vs_consensus": metrics.upside,
            },
            ensure_ascii=False,
        )

        raw = llm.complete_json(system, user)

        # Składamy raport: część jakościowa od LLM, METRYKI z naszych obliczeń.
        report = Report(
            recommendation=str(raw.get("recommendation", "Neutralna")),
            rationale=str(raw.get("rationale", "")),
            risks=[str(r) for r in raw.get("risks", [])],
            metrics=metrics,
        )
        return {"report": report}

    return synthesize_node
