"""Embeddingi: zamiana tekstu na wektory.

Abstrakcja Embedder (jak LLMClient) — niezależność i testowalność:
  • SentenceTransformerEmbedder — prawdziwy model lokalny (offline, bez kosztów API),
  • FakeEmbedder — deterministyczna atrapa do testów, bez pobierania modelu.

WAŻNE: oba embeddery produkują wektor o tym samym wymiarze (EMBED_DIM),
żeby schemat tabeli w pgvector (vector(EMBED_DIM)) pasował niezależnie od wyboru.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

# Wymiar wektora. 384 = wymiar all-MiniLM-L6-v2 (mały, szybki, dobry domyślny model).
EMBED_DIM = 384

# Domyślny model lokalny. Mały (~80 MB), szybki, sensowna jakość do newsów.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(ABC):
    """Wspólny interfejs embeddera."""

    dim: int = EMBED_DIM

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Zwraca listę wektorów (po jednym na każdy tekst)."""
        ...


class SentenceTransformerEmbedder(Embedder):
    """Lokalny model z biblioteki sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        # Import i załadowanie modelu leniwie — biblioteka jest ciężka (torch),
        # nie chcemy jej wymagać, gdy używamy atrapy.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


class FakeEmbedder(Embedder):
    """Deterministyczna atrapa: hashuje tekst w powtarzalny, znormalizowany wektor.

    Nie oddaje znaczenia semantycznego, ale jest STABILNA i bez zależności —
    pozwala przetestować cały tor RAG (ingest -> pgvector -> search) bez torcha.
    """

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

    "fake" => atrapa. W przeciwnym razie próbujemy modelu lokalnego, a gdy
    biblioteka/torch są niedostępne, schodzimy do atrapy zamiast wywalać start.
    """
    if provider == "fake":
        return FakeEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        return FakeEmbedder()
