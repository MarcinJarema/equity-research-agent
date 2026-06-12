"""Interfejs do bazy wektorowej (PostgreSQL + pgvector).

Trzyma kawałki tekstu (chunki) z embeddingami i pozwala je wyszukiwać po
podobieństwie kosinusowym. Filtrowanie po tickerze => kontekst dotyczy WŁAŚCIWEJ spółki.

DLACZEGO pgvector, a nie dedykowana baza (Qdrant):
  • jedna baza na wszystko (mniej ruchomych części w utrzymaniu),
  • Postgres i tak znamy/mamy — pgvector to tylko rozszerzenie,
  • na skali projektu portfolio wydajność w zupełności wystarcza.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from app.rag.embeddings import EMBED_DIM


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


class VectorStore:
    """Cienka warstwa nad tabelą pgvector."""

    def __init__(self, dsn: str):
        self._dsn = dsn

    def _connect_raw(self) -> psycopg.Connection:
        """Surowe połączenie — bez rejestracji typu vector.

        Używane przez init_schema: na świeżej bazie typ 'vector' jeszcze nie
        istnieje, więc register_vector by się wywalił. DDL go nie wymaga.
        """
        return psycopg.connect(self._dsn)

    def _connect(self) -> psycopg.Connection:
        conn = self._connect_raw()
        register_vector(conn)  # uczy psycopg typu vector <-> list[float]
        return conn

    def init_schema(self) -> None:
        """Tworzy rozszerzenie, tabelę i indeks (idempotentnie)."""
        with self._connect_raw() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id        BIGSERIAL PRIMARY KEY,
                    ticker    TEXT NOT NULL,
                    source    TEXT NOT NULL,
                    content   TEXT NOT NULL,
                    embedding vector({EMBED_DIM}) NOT NULL
                )
                """
            )
            # Indeks pod podobieństwo kosinusowe. ivfflat jest lekki i wystarcza;
            # 'lists' to liczba kubełków — mała wartość OK dla małego zbioru.
            cur.execute(
                "CREATE INDEX IF NOT EXISTS documents_embedding_idx "
                "ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS documents_ticker_idx ON documents (ticker)")
            conn.commit()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Wstawia chunki. Zwraca liczbę dodanych wierszy."""
        if not chunks:
            return 0
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO documents (ticker, source, content, embedding) "
                "VALUES (%s, %s, %s, %s)",
                [(c.ticker, c.source, c.content, c.embedding) for c in chunks],
            )
            conn.commit()
            return len(chunks)

    def has_documents(self, ticker: str) -> bool:
        """Czy mamy JAKIKOLWIEK kontekst dla tickera? Decyduje, czy włączać RAG.

        Każdy błąd (baza nieosiągalna, brak schematu) traktujemy jak 'brak' —
        agent ma działać dalej bez RAG, a nie wywalać się.
        """
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1 FROM documents WHERE ticker = %s LIMIT 1", (ticker,))
                return cur.fetchone() is not None
        except Exception:
            return False

    def search(self, ticker: str, query_embedding: list[float], k: int = 4) -> list[SearchHit]:
        """Zwraca k najbliższych chunków dla tickera (kosinus: operator <=>)."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, source, embedding <=> %s AS distance
                FROM documents
                WHERE ticker = %s
                ORDER BY distance
                LIMIT %s
                """,
                (query_embedding, ticker, k),
            )
            return [SearchHit(content=r[0], source=r[1], distance=float(r[2])) for r in cur.fetchall()]
