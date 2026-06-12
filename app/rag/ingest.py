"""Ingest RAG: newsy spółki -> chunki -> embeddingi -> pgvector.

STRATEGIA CHUNKINGU (to jest pytane na rozmowie):

Naiwny chunking = tnij tekst co N znaków. Czemu zawodzi:
  • przecina zdania i fakty w pół — "cena docelowa wzrosła do" | "180 USD" trafiają
    do dwóch chunków, więc żaden nie jest sam w sobie wyszukiwalny;
  • miesza dwa newsy w jednym chunku, rozmywając sygnał semantyczny;
  • ignoruje strukturę dokumentu.

Nasza strategia (świadoma, dopasowana do krótkich newsów finansowych):
  1. Każdy news to naturalna jednostka — NIE sklejamy różnych newsów.
  2. W obrębie newsa dzielimy po ZDANIACH (granica semantyczna), nie po znakach.
  3. Zdania grupujemy do chunku ~docelowej długości, z NAKŁADKĄ (overlap) jednego
     zdania — żeby fakt na styku dwóch chunków nie wypadł z kontekstu.
  4. Do treści doklejamy tytuł newsa jako prefiks — chunk jest samoopisowy
     (wie, czego dotyczy, nawet wyrwany z kontekstu).
"""

from __future__ import annotations

import re

from app.rag.embeddings import Embedder
from app.rag.store import Chunk, VectorStore
from app.tools.news import get_news

# Docelowy rozmiar chunku w znakach (newsy są krótkie — duże chunki nie mają sensu).
TARGET_CHARS = 500
SENTENCE_OVERLAP = 1  # ile zdań nakładki między sąsiednimi chunkami


def split_sentences(text: str) -> list[str]:
    """Prosty podział na zdania po . ! ? + nowych liniach. Bez ciężkich zależności."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, *, target_chars: int = TARGET_CHARS) -> list[str]:
    """Grupuje zdania w chunki ~target_chars z nakładką SENTENCE_OVERLAP zdań."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for sent in sentences:
        if current and length + len(sent) > target_chars:
            chunks.append(" ".join(current))
            # Nakładka: przenosimy ostatnie SENTENCE_OVERLAP zdań do nowego chunku.
            current = current[-SENTENCE_OVERLAP:] if SENTENCE_OVERLAP else []
            length = sum(len(s) for s in current)
        current.append(sent)
        length += len(sent)

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(ticker: str, embedder: Embedder) -> list[Chunk]:
    """Pobiera newsy, tnie je na chunki i liczy embeddingi."""
    news = get_news(ticker)
    pending: list[tuple[str, str]] = []  # (source, content)

    for item in news:
        body = item.summary or item.title
        # Tytuł jako prefiks => chunk jest samoopisowy.
        prefixed = f"{item.title}. {body}" if item.summary else item.title
        for piece in chunk_text(prefixed):
            pending.append((item.title, piece))

    if not pending:
        return []

    embeddings = embedder.embed([content for _, content in pending])
    return [
        Chunk(ticker=ticker.upper(), source=source, content=content, embedding=emb)
        for (source, content), emb in zip(pending, embeddings, strict=True)
    ]


def ingest_ticker(ticker: str, store: VectorStore, embedder: Embedder) -> int:
    """Pełny ingest dla tickera: schemat -> chunki -> zapis. Zwraca liczbę chunków."""
    # Wymiar tabeli bierzemy z embeddera => tabela zawsze pasuje do modelu.
    store.init_schema(embedder.dim)
    chunks = build_chunks(ticker, embedder)
    return store.add_chunks(chunks)


def _main() -> None:
    """CLI: `python -m app.rag.ingest TICKER [TICKER ...]` — ładuje newsy do bazy."""
    import sys

    from app.core.config import get_settings
    from app.rag.embeddings import get_embedder

    tickers = [t.upper() for t in sys.argv[1:]]
    if not tickers:
        print("Użycie: python -m app.rag.ingest TICKER [TICKER ...]")
        raise SystemExit(1)

    settings = get_settings()
    store = VectorStore(settings.database_url)
    embedder = get_embedder(settings.embed_provider)

    try:
        for ticker in tickers:
            n = ingest_ticker(ticker, store, embedder)
            print(f"{ticker}: zapisano {n} chunków.")
    finally:
        store.close()  # zamknij pulę (wątek tła) przed wyjściem


if __name__ == "__main__":
    _main()
