"""Stan agenta + schemat raportu (structured output).

JAK DZIAŁA STAN W LANGGRAPH (to jest sedno do zrozumienia):
  • Stan to jeden słownik (tu: TypedDict AgentState) wędrujący przez graf.
  • Każdy WĘZEŁ dostaje aktualny stan i zwraca słownik z polami, które chce
    ZAKTUALIZOWAĆ — LangGraph scala (merge) ten wynik ze stanem.
  • Węzeł NIE musi zwracać całego stanu, tylko swoją "działkę".
    Np. węzeł market_data zwraca {"market_data": ...}; reszta pól zostaje.
  • Kolejny węzeł widzi już zaktualizowany stan. Tak płynie informacja:
    router ustawia plan -> market_data dokłada dane -> metrics dokłada metryki
    -> synthesize czyta wszystko i dokłada gotowy raport.
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field

from app.tools.market_data import MarketData
from app.tools.metrics import Metrics
from app.tools.news import NewsItem

DISCLAIMER = (
    "To narzędzie analityczne i projekt edukacyjny, NIE porada inwestycyjna. "
    "Dane mogą być niekompletne, a wnioski błędne. Nie podejmuj decyzji "
    "inwestycyjnych na tej podstawie."
)


class Report(BaseModel):
    """Ustrukturyzowany raport — wyjście agenta.

    WAŻNE: część jakościową (recommendation/rationale/risks) generuje LLM,
    ale LICZBY (metrics) wstawiamy z własnych obliczeń, NIE z modelu —
    żeby model nie halucynował wartości metryk. To świadoma decyzja anty-halucynacyjna.
    """

    recommendation: str = Field(..., description="np. Pozytywna / Neutralna / Negatywna")
    rationale: str = Field(..., description="Uzasadnienie oparte o metryki i kontekst")
    risks: list[str] = Field(default_factory=list, description="Główne ryzyka")
    metrics: Metrics
    # Tytuły newsów wykorzystanych jako kontekst — transparentność źródeł.
    news_used: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


class AgentState(TypedDict, total=False):
    """Stan przepływający przez graf. total=False => pola dochodzą stopniowo."""

    ticker: str                 # wejście
    plan: list[str]             # które narzędzia odpalić (ustawia router)
    market_data: MarketData     # dokłada węzeł market_data
    metrics: Metrics            # dokłada węzeł metrics
    news: list[NewsItem]        # dokłada węzeł news
    rag_context: list[str]      # dokłada węzeł rag (kontekst z bazy wektorowej)
    report: Report              # dokłada węzeł synthesize (wyjście)
    error: str                  # ustawiane, gdy coś pójdzie nie tak
