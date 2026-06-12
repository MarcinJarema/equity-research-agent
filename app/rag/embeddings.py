"""Embeddingi: zamiana tekstu na wektory.

Abstrakcja Embedder (jak LLMClient) — niezależność i testowalność:
  • SentenceTransformerEmbedder — prawdziwy model lokalny (offline, bez kosztów API);
    model ładowany LENIWIE przy pierwszym użyciu, żeby nie blokować startu aplikacji,
  • FakeEmbedder — deterministyczna atrapa do testów, bez pobierania modelu.

Wymiar wektora bierzemy z faktycznego embeddera (Embedder.dim), a nie ze stałej —
dzięki temu tabela w pgvector ma wymiar zgodny z modelem, niezależnie od wyboru.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Wymiar atrapy = wymiar all-MiniLM-L6-v2 (mały, szybki, dobry domyślny model),
# żeby przełączenie fake <-> realny model nie zmieniało schematu tabeli.
EMBED_DIM = 384

# Domyślny model lokalny. Mały (~80 MB), szybki, sensowna jakość do newsów.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(ABC):
    """Wspólny interfejs embeddera."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Wymiar zwracanych wektorów (do schematu tabeli pgvector)."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Zwraca listę wektorów (po jednym na każdy tekst)."""
        ...


class SentenceTransformerEmbedder(Embedder):
    """Lokalny model z biblioteki sentence-transformers (ładowany leniwie)."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None  # ładowane przy pierwszym użyciu (_ensure)

    def _ensure(self):
        if self._model is None:
            # Import i załadowanie modelu dopiero teraz — biblioteka jest ciężka (torch).
            from sentence_transformers import SentenceTransformer

            logger.info("Ładuję model embeddingów: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self._ensure().get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._ensure().encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


class FakeEmbedder(Embedder):
    """Deterministyczna atrapa: hashuje tekst w powtarzalny, znormalizowany wektor.

    Nie oddaje znaczenia semantycznego, ale jest STABILNA i bez zależności —
    pozwala przetestować cały tor RAG (ingest -> pgvector -> search) bez torcha.
    """

    @property
    def dim(self) -> int:
        return EMBED_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vector(t) for t in texts]

    def _hash_vector(self, text: str) -> list[float]:
        # Rozsiewamy bajty hashy po wymiarach, potem normalizujemy do długości 1.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(EMBED_DIM)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


def get_embedder(provider: str = "sentence-transformers") -> Embedder:
    """Fabryka embeddera wg konfiguracji.

    "fake" => atrapa. W przeciwnym razie używamy modelu lokalnego, ale tylko gdy
    biblioteka jest zainstalowana (sprawdzamy TANIO, bez ładowania modelu); inaczej
    logujemy i schodzimy do atrapy zamiast wywalać aplikację.
    """
    if provider == "fake":
        return FakeEmbedder()
    if importlib.util.find_spec("sentence_transformers") is None:
        logger.warning(
            "Brak sentence-transformers — używam FakeEmbedder (RAG bez znaczenia semantycznego)."
        )
        return FakeEmbedder()
    return SentenceTransformerEmbedder()
