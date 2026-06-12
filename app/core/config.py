"""Konfiguracja aplikacji + limity kosztów.

Używamy pydantic-settings: wartości czytane są z pliku .env (lub zmiennych
środowiskowych). Dzięki temu sekrety (klucze API) NIE są w kodzie, a cała
konfiguracja jest w jednym, walidowanym miejscu.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralna konfiguracja. Pola mapują się 1:1 na klucze z .env (case-insensitive)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # nieznane klucze w .env nie wywalają aplikacji
    )

    # --- Wybór dostawcy LLM (abstrakcja LLMClient — pełna implementacja w ETAPIE 2) ---
    llm_provider: Literal["groq", "openai", "anthropic"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"

    # Klucze API. Domyślnie puste — w ETAPIE 1 LLM jest mockowany, więc nie są wymagane.
    groq_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- Kontrola kosztów (świadomy element: limit + niska temperatura) ---
    max_tokens: int = 1024
    llm_temperature: float = 0.2

    # --- Observability (ETAP 4) ---
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "equity-research-agent"

    # --- RAG / pgvector (ETAP 3) ---
    database_url: str = "postgresql://postgres:postgres@db:5432/equity"


@lru_cache
def get_settings() -> Settings:
    """Zwraca singleton ustawień.

    lru_cache => plik .env czytany jest raz na proces, nie przy każdym żądaniu.
    W FastAPI używamy tego jako zależności (Depends) lub wołamy bezpośrednio.
    """
    return Settings()
