"""Definicja i kompilacja grafu LangGraph: router -> tools -> synthesize.

Topologia (ETAP 3):

    START -> router -> market_data --(błąd?)--> END
                            │
                          (ok)
                            ▼
                         metrics -> news -> [RAG?] --(brak danych)--> synthesize -> END
                                              │
                                            (są dane)
                                              ▼
                                             rag --------------------> synthesize -> END

Dwie conditional edges pokazują, jak agent ROZGAŁĘZIA przepływ na podstawie stanu:
  1. po market_data — obsługa błędu (pomijamy resztę),
  2. po news — REALNA decyzja: użyć RAG tylko, gdy w bazie jest kontekst dla tickera.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    make_rag_node,
    make_should_use_rag,
    make_synthesize_node,
    market_data_node,
    metrics_node,
    news_node,
    router_node,
)
from app.agent.state import AgentState
from app.llm.client import LLMClient
from app.rag.embeddings import Embedder
from app.rag.store import VectorStore


def _after_market_data(state: AgentState) -> str:
    """Conditional edge: dokąd iść po pobraniu danych."""
    return "end" if state.get("error") else "ok"


def build_graph(llm: LLMClient, store: VectorStore, embedder: Embedder):
    """Buduje i kompiluje graf. Zależności (LLM, store, embedder) wstrzykujemy."""
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("market_data", market_data_node)
    graph.add_node("metrics", metrics_node)
    graph.add_node("news", news_node)
    graph.add_node("rag", make_rag_node(store, embedder))
    graph.add_node("synthesize", make_synthesize_node(llm))

    graph.add_edge(START, "router")
    graph.add_edge("router", "market_data")
    graph.add_conditional_edges(
        "market_data",
        _after_market_data,
        {"ok": "metrics", "end": END},
    )
    graph.add_edge("metrics", "news")
    # Decyzja o RAG: 'rag' (jest kontekst) albo wprost 'synthesize'.
    graph.add_conditional_edges(
        "news",
        make_should_use_rag(store),
        {"rag": "rag", "synthesize": "synthesize"},
    )
    graph.add_edge("rag", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
