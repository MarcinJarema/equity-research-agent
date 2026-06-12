"""Interfejs do bazy wektorowej (PostgreSQL + pgvector).

Trzyma kawałki tekstu (chunki) z embeddingami i pozwala je wyszukiwać po
podobieństwie kosinusowym. Filtrowanie po tickerze => kontekst dotyczy WŁAŚCIWEJ spółki.

Połączenia idą przez PULĘ (psycopg_pool) — jedno żądanie /analyze odpytuje bazę
dwa razy (has_documents + search), a pula reużywa połączeń zamiast otwierać nowe
TCP za każdym razem. Pulę tworzymy LENIWIE (przy pierwszym użyciu), więc start
aplikacji nie zależy od dostępności bazy.

DLACZEGO pgvector, a nie dedykowana baza (Qdrant):
  • jedna baza na wszystko (mniej ruchomych części w utrzymaniu),
  • Postgres i tak znamy/mamy — pgvector to tylko rozszerzenie,
  • na skali projektu portfolio wydajność w zupełności wystarcza.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Kawałek dokumentu zapisany w bazie (wejście do ingestu)."""

    ticker: str
    source: str          # skąd pochodzi (np. tytuł/serwis newsowy)
    content: str         # treść chunku
    embedding: list[float]


@dataclass
class SearchHit:
    """Wynik wyszukiwania: treść + dystans (mniejszy = bliżej)."""

    content: str
    source: str
    distance: float


def _configure(conn: psycopg.Connection) -> None:
    """Uruchamiane na KAŻDYM nowym połączeniu z puli.

    Gwarantuje rozszerzenie vector i rejestruje typ vector<->list[float], więc
    każde pobrane z puli połączenie od razu umie obsłużyć embeddingi. CREATE
    EXTENSION jest idempotentne, a połączeń jest mało (pula), więc koszt znikomy.
    """
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)


class VectorStore:
    """Cienka warstwa nad tabelą pgvector, z pulą połączeń."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: ConnectionPool | None = None

    def _get_pool(self) -> ConnectionPool:
        """Tworzy pulę przy pierwszym użyciu (nie przy starcie aplikacji)."""
        if self._pool is None:
            pool = ConnectionPool(
                self._dsn,
                min_size=1,
                max_size=5,
                timeout=5,                          # max czekania na połączenie z puli
                kwargs={"connect_timeout": 5},      # szybki fail, gdy baza nieosiągalna
                configure=_configure,
                open=False,
            )
            pool.open(wait=False)  # nie blokuj na starcie — połączenia powstają w tle
            self._pool = pool
        return self._pool

    def close(self) -> None:
        """Zamyka pulę (wątek tła). Wołać przy zatrzymaniu aplikacji/skryptu."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def init_schema(self, dim: int) -> None:
        """Tworzy tabelę i indeks (idempotentnie). Wymiar wektora z embeddera."""
        with self._get_pool().connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id        BIGSERIAL PRIMARY KEY,
                    ticker    TEXT NOT NULL,
                    source    TEXT NOT NULL,
                    content   TEXT NOT NULL,
                    embedding vector({dim}) NOT NULL
                )
                """
            )
            # Indeks pod podobieństwo kosinusowe (ivfflat, lekki dla małego zbioru).
            conn.execute(
                "CREATE INDEX IF NOT EXISTS documents_embedding_idx "
                "ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS documents_ticker_idx ON documents (ticker)")
            conn.commit()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Wstawia chunki. Zwraca liczbę dodanych wierszy."""
        if not chunks:
            return 0
        with self._get_pool().connection() as conn:
            conn.cursor().executemany(
                "INSERT INTO documents (ticker, source, content, embedding) "
                "VALUES (%s, %s, %s, %s)",
                [(c.ticker, c.source, c.content, c.embedding) for c in chunks],
            )
            conn.commit()
            return len(chunks)

    def has_documents(self, ticker: str) -> bool:
        """Czy mamy JAKIKOLWIEK kontekst dla tickera? Decyduje, czy włączać RAG.

        Błąd (baza nieosiągalna, brak schematu) logujemy i traktujemy jak 'brak' —
        agent ma działać dalej bez RAG, a nie wywalać się.
        """
        try:
            with self._get_pool().connection() as conn:
                cur = conn.execute("SELECT 1 FROM documents WHERE ticker = %s LIMIT 1", (ticker,))
                return cur.fetchone() is not None
        except Exception as exc:
            logger.warning("has_documents(%s) niedostępne, pomijam RAG: %s", ticker, exc)
            return False

    def search(self, ticker: str, query_embedding: list[float], k: int = 4) -> list[SearchHit]:
        """Zwraca k najbliższych chunków dla tickera (kosinus: operator <=>)."""
        with self._get_pool().connection() as conn:
            # %s::vector — bez jawnego rzutowania psycopg wysyła listę jako
            # double precision[], a operator <=> wymaga typu vector.
            cur = conn.execute(
                """
                SELECT content, source, embedding <=> %s::vector AS distance
                FROM documents
                WHERE ticker = %s
                ORDER BY distance
                LIMIT %s
                """,
                (query_embedding, ticker, k),
            )
            return [
                SearchHit(content=r[0], source=r[1], distance=float(r[2]))
                for r in cur.fetchall()
            ]
