"""FastAPI — punkt wejścia aplikacji.

ETAP 1: /health + /analyze jako PROSTY przepływ:
    ticker -> get_market_data -> calc_metrics -> (mock LLM) -> raport.
Agenta LangGraph w miejsce mocka wprowadzimy w ETAPIE 2.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.tools.market_data import MarketDataError, get_market_data
from app.tools.metrics import Metrics, calc_metrics

DISCLAIMER = (
    "To narzędzie analityczne i projekt edukacyjny, NIE porada inwestycyjna. "
    "Dane mogą być niekompletne, a wnioski błędne. Nie podejmuj decyzji "
    "inwestycyjnych na tej podstawie."
)

app = FastAPI(
    title="Equity Research Agent",
    description="Agentowy system analizy spółek giełdowych (ETAP 1: szkielet bez agenta).",
    version="0.1.0",
)


# --- Schematy wejścia/wyjścia (kontrakt API) ---

class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"], description="Symbol spółki, np. AAPL.")


class AnalyzeResponse(BaseModel):
    """Ustrukturyzowany raport. W ETAPIE 1 'summary' generuje mock, nie LLM."""

    ticker: str
    current_price: float
    currency: str | None
    metrics: Metrics
    summary: str
    disclaimer: str = DISCLAIMER


# --- Endpointy ---

@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — używane m.in. przez docker-compose / orchestrację."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Analizuje spółkę: dane rynkowe -> metryki -> (mock) synteza."""
    try:
        data = get_market_data(req.ticker)
    except MarketDataError as exc:
        # 422: ticker formalnie poprawny, ale brak dla niego danych.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    metrics = calc_metrics(data)
    summary = _mock_synthesize(req.ticker.upper(), data.current_price, metrics)

    return AnalyzeResponse(
        ticker=data.ticker,
        current_price=data.current_price,
        currency=data.currency,
        metrics=metrics,
        summary=summary,
    )


def _mock_synthesize(ticker: str, price: float, metrics: Metrics) -> str:
    """ZAŚLEPKA LLM (ETAP 1).

    Buduje deterministyczny opis z policzonych metryk. W ETAPIE 2 zastąpi to
    węzeł 'synthesize' agenta LangGraph wołający prawdziwy LLM (structured output).
    Tu celowo NIE ma żadnej "rekomendacji" — to wymaga modelu, nie if-a.
    """
    parts = [f"Analiza {ticker} (cena bieżąca: {price:.2f})."]

    if metrics.momentum_12_1 is not None:
        kierunek = "dodatnie" if metrics.momentum_12_1 >= 0 else "ujemne"
        parts.append(f"Momentum 12-1: {metrics.momentum_12_1:+.1%} ({kierunek}).")
    else:
        parts.append("Momentum 12-1: brak (za krótka historia).")

    if metrics.upside is not None:
        parts.append(f"Upside vs konsensus analityków: {metrics.upside:+.1%}.")
    else:
        parts.append("Upside: brak ceny docelowej konsensusu.")

    parts.append("[Synteza LLM zostanie dodana w ETAPIE 2.]")
    return " ".join(parts)
