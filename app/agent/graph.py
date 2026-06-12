"""Definicja i kompilacja grafu LangGraph: router -> tools -> synthesize.

Topologia (ETAP 2, MVP):

    START -> router -> market_data --(błąd?)--> END
                            │
                          (ok)
                            ▼
                         metrics -> synthesize -> END

Conditional edge po market_data pokazuje, jak LangGraph ROZGAŁĘZIA przepływ
na podstawie stanu (tu: jeśli pobranie danych się nie udało, pomijamy resztę).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    make_synthesize_node,
    market_data_node,
    metrics_node,
    router_node,
)
from app.agent.state import AgentState
from app.llm.client import LLMClient


def _after_market_data(state: AgentState) -> str:
    """Funkcja decyzyjna conditional edge: dokąd iść po pobraniu danych."""
    return "end" if state.get("error") else "ok"


def build_graph(llm: LLMClient):
    """Buduje i kompiluje graf. Klient LLM wstrzykujemy (prawdziwy lub atrapa)."""
    graph = StateGraph(AgentState)

    # Rejestrujemy węzły (nazwa -> funkcja).
    graph.add_node("router", router_node)
    graph.add_node("market_data", market_data_node)
    graph.add_node("metrics", metrics_node)
    graph.add_node("synthesize", make_synthesize_node(llm))

    # Krawędzie = kolejność przepływu stanu.
    graph.add_edge(START, "router")
    graph.add_edge("router", "market_data")
    graph.add_conditional_edges(
        "market_data",
        _after_market_data,
        {"ok": "metrics", "end": END},  # mapowanie etykiety -> węzeł docelowy
    )
    graph.add_edge("metrics", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
