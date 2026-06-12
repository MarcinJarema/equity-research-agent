"""Abstrakcja dostawcy LLM (interfejs LLMClient) + implementacje.

DLACZEGO warstwa abstrakcji, a nie wołanie Groqa wprost:
  • niezależność od dostawcy — zmiana Groq -> OpenAI/Anthropic = jedna implementacja,
    reszta kodu (agent) nic nie wie o tym, kto liczy;
  • testowalność — w testach wstrzykujemy FakeLLMClient, bez sieci i bez kosztów.

Kontrakt jest wąski: complete_json(system, user) -> dict.
Wymuszamy JSON, bo agent potrzebuje STRUCTURED OUTPUT, nie wolnego tekstu.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from app.core.config import Settings


class LLMClient(ABC):
    """Wspólny interfejs każdego dostawcy LLM."""

    @abstractmethod
    def complete_json(self, system: str, user: str) -> dict:
        """Zwraca odpowiedź modelu sparsowaną jako obiekt JSON (dict)."""
        ...


class GroqClient(LLMClient):
    """Implementacja na Groq (API kompatybilne z OpenAI, tryb JSON)."""

    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        # Import lokalny: pakiet groq potrzebny tylko, gdy realnie używamy Groqa.
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete_json(self, system: str, user: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # response_format wymusza poprawny JSON po stronie API.
            response_format={"type": "json_object"},
            max_tokens=self._max_tokens,      # twardy limit kosztów
            temperature=self._temperature,    # niska => stabilniejszy, mniej "kreatywny"
            timeout=30.0,                      # nie wieszamy workera na zawieszonym połączeniu
        )
        content = resp.choices[0].message.content
        if not content:
            return {}  # pusta odpowiedź modelu — synteza użyje wartości domyślnych
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM zwrócił niepoprawny JSON: {content[:200]!r}") from exc


class FakeLLMClient(LLMClient):
    """Atrapa do testów i trybu bez klucza.

    Nie woła sieci — zwraca deterministyczny, sensowny raport. Dzięki temu
    cały graf agenta testujemy bez kosztów i bez zależności od dostępności API.
    """

    def complete_json(self, system: str, user: str) -> dict:
        return {
            "recommendation": "Neutralna",
            "rationale": (
                "Atrapa LLM (brak klucza lub tryb testowy). To miejsce, w którym "
                "prawdziwy model oceniłby metryki i kontekst."
            ),
            "risks": ["Ocena pochodzi z atrapy, nie z modelu."],
        }


def get_llm_client(settings: Settings) -> LLMClient:
    """Fabryka: zwraca klienta wg LLM_PROVIDER z konfiguracji.

    Jeśli wybrany dostawca nie ma klucza, świadomie schodzimy do FakeLLMClient —
    aplikacja startuje i działa (z jawnie atrapowaną syntezą), zamiast się wywalać.
    """
    provider = settings.llm_provider

    if provider == "groq":
        if not settings.groq_api_key:
            return FakeLLMClient()
        return GroqClient(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            max_tokens=settings.max_tokens,
            temperature=settings.llm_temperature,
        )

    # Miejsce na kolejnych dostawców — utrzymujemy "niezależność od dostawcy".
    if provider in ("openai", "anthropic"):
        raise NotImplementedError(
            f"Dostawca '{provider}' jeszcze nie zaimplementowany. "
            "Interfejs jest gotowy — dołóż klasę {Provider}Client wzorem GroqClient."
        )

    raise ValueError(f"Nieznany LLM_PROVIDER: {provider}")
