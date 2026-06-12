"""Observability: trace przepływu agenta w LangSmith.

JAK TO DZIAŁA:
LangGraph integruje się z callbackami LangChain. Gdy ustawimy odpowiednie zmienne
środowiskowe (klucz + flaga), KAŻDE wywołanie skompilowanego grafu jest
automatycznie śledzone w LangSmith — widać przepływ węzeł po węźle
(router -> market_data -> metrics -> news -> rag -> synthesize), czasy i wejścia/wyjścia.

DLACZEGO opcjonalne:
trace ma sens tylko z kluczem LangSmith. Bez klucza schodzimy cicho do trybu
wyłączonego — projekt da się sklonować i uruchomić bez konta LangSmith.
"""

from __future__ import annotations

import os

from app.core.config import Settings


def setup_observability(settings: Settings) -> bool:
    """Konfiguruje LangSmith, jeśli włączony i jest klucz. Zwraca: czy aktywny.

    Ustawiamy zmienne, których szuka LangChain/LangGraph. Robimy to RAZ przy
    starcie aplikacji, zanim zbudujemy graf.
    """
    if not settings.langsmith_tracing:
        return False
    if not settings.langsmith_api_key:
        # Flaga włączona, ale brak klucza — nie udajemy, że trace działa.
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    # Starsze nazwy (LANGCHAIN_*) dla kompatybilności wstecznej bibliotek.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    return True
