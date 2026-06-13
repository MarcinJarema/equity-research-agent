"""FastAPI — punkt wejścia aplikacji.

ETAP 2: /analyze uruchamia agenta LangGraph (router -> tools -> synthesize).
Graf budujemy RAZ przy starcie (trzyma wstrzykniętego klienta LLM), a każde
żądanie to graf.invoke({"ticker": ...}).
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.agent.graph import build_graph
from app.agent.state import Report
from app.core.config import get_settings
from app.core.observability import setup_observability
from app.llm.client import get_llm_client
from app.rag.embeddings import get_embedder
from app.rag.store import VectorStore
from app.tools.market_data import MarketDataError, get_market_data
from app.tools.metrics import calc_metrics
from app.tools.rating import classify_rating
from app.tools.revisions import get_revision_trend

# Budujemy agenta raz na proces. Zależności wybierane z .env; fabryki schodzą
# do atrap (brak klucza LLM / brak modelu embeddingów), więc aplikacja zawsze wstaje.
_settings = get_settings()

# Trace przepływu (LangSmith) — aktywny tylko z flagą + kluczem. Musi być PRZED
# zbudowaniem grafu, bo to moment, w którym LangGraph podpina callbacki tracingu.
TRACING_ENABLED = setup_observability(_settings)

_store = VectorStore(_settings.database_url)
_agent = build_graph(
    llm=get_llm_client(_settings),
    store=_store,
    embedder=get_embedder(_settings.embed_provider),
)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Cykl życia aplikacji — przy zatrzymaniu zamykamy pulę połączeń."""
    yield
    _store.close()


app = FastAPI(
    title="Equity Research Agent",
    description="Agentowy system analizy spółek giełdowych (LangGraph + tool-calling).",
    version="0.3.0",
    lifespan=_lifespan,
)


# Dozwolony format tickera: litery/cyfry + kropka/myślnik (np. BRK.B, RDS-A), do 10 znaków.
# Walidacja na wejściu zawęża, co trafia dalej do yfinance, bazy i promptu LLM.
_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,10}$")

# Domyślne uniwersum do porównania (zróżnicowane sektory). UI pozwala wybrać podzbiór.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA",
    "JPM", "XOM", "KO", "PFE", "WMT", "DIS",
]


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"], description="Symbol spółki, np. AAPL.")

    @field_validator("ticker")
    @classmethod
    def _validate_ticker(cls, v: str) -> str:
        v = v.strip()
        if not _TICKER_RE.match(v):
            raise ValueError(
                "Nieprawidłowy ticker (dozwolone: litery, cyfry, '.', '-', do 10 znaków)."
            )
        return v.upper()


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe. Pokazuje też, czy trace LangSmith jest aktywny."""
    return {"status": "ok", "tracing": TRACING_ENABLED}


@app.post("/analyze", response_model=Report)
def analyze(req: AnalyzeRequest) -> Report:
    """Uruchamia agenta na podanym tickerze i zwraca ustrukturyzowany raport."""
    try:
        result = _agent.invoke({"ticker": req.ticker})
    except Exception as exc:
        # Nieoczekiwany błąd w którymś węźle (yfinance, LLM, baza) — nie wyciekamy
        # stack trace'a; zwracamy czytelny 502 (błąd źródła danych/LLM).
        raise HTTPException(status_code=502, detail=f"Błąd podczas analizy: {exc}") from exc

    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])

    report = result.get("report")
    if report is None:
        raise HTTPException(status_code=502, detail="Agent nie zwrócił raportu.")
    return report


# --- Porównanie wielu spółek (regułowy rating, bez LLM => szybkie i tanie) ---

class CompareRequest(BaseModel):
    tickers: list[str] = Field(
        default_factory=list,
        description="Lista tickerów. Puste => domyślne uniwersum (/universe).",
    )


class CompareRow(BaseModel):
    ticker: str
    rating: str | None = None
    momentum_12_1: float | None = None
    upside: float | None = None
    revisions_negative_2m: bool = False
    error: str | None = None


@app.get("/universe")
def universe() -> list[str]:
    """Domyślna lista spółek do porównania (do zasilenia UI)."""
    return DEFAULT_UNIVERSE


@app.post("/compare", response_model=list[CompareRow])
def compare(req: CompareRequest) -> list[CompareRow]:
    """Liczy regułowy rating (Strong Buy/Buy/Hold/Sell) dla wielu spółek naraz."""
    tickers = req.tickers or DEFAULT_UNIVERSE
    rows: list[CompareRow] = []

    for raw in tickers:
        ticker = raw.strip().upper()
        if not _TICKER_RE.match(ticker):
            rows.append(CompareRow(ticker=ticker, error="nieprawidłowy ticker"))
            continue
        try:
            data = get_market_data(ticker)
            metrics = calc_metrics(data)
            trend = get_revision_trend(ticker)
            rating = classify_rating(metrics.upside, metrics.momentum_12_1, trend.negative_2m)
            rows.append(
                CompareRow(
                    ticker=ticker,
                    rating=rating,
                    momentum_12_1=metrics.momentum_12_1,
                    upside=metrics.upside,
                    revisions_negative_2m=trend.negative_2m,
                )
            )
        except MarketDataError as exc:
            rows.append(CompareRow(ticker=ticker, error=str(exc)))
        except Exception as exc:  # noqa: BLE001 — pojedyncza spółka nie wywala całego porównania
            rows.append(CompareRow(ticker=ticker, error=f"błąd: {exc}"))

    return rows
