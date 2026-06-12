"""Narzędzie rag_search: wyszukuje kontekst dla spółki w bazie wektorowej.

To kolejne narzędzie agenta (obok market_data/metrics/news). Agent używa go,
gdy w bazie jest już zingestowany kontekst dla danego tickera (patrz router).
"""

from __future__ import annotations

import logging

from app.rag.embeddings import Embedder
from app.rag.store import VectorStore

logger = logging.getLogger(__name__)


def rag_search(
    ticker: str,
    query: str,
    store: VectorStore,
    embedder: Embedder,
    *,
    k: int = 4,
) -> list[str]:
    """Embeduje zapytanie i zwraca treść k najbliższych chunków dla tickera.

    Przy braku danych/błędzie bazy zwraca pustą listę — agent działa dalej bez RAG.
    """
    try:
        query_vec = embedder.embed([query])[0]
        hits = store.search(ticker.upper(), query_vec, k=k)
        return [h.content for h in hits]
    except Exception as exc:
        logger.warning("rag_search(%s) nieudane, kontynuuję bez RAG: %s", ticker, exc)
        return []
