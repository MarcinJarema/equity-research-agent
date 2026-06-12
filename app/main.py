"""FastAPI — punkt wejścia aplikacji.

ETAP 2: /analyze uruchamia agenta LangGraph (router -> tools -> synthesize).
Graf budujemy RAZ przy starcie (trzyma wstrzykniętego klienta LLM), a każde
żądanie to graf.invoke({"ticker": ...}).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import build_graph
from app.agent.state import Report
from app.core.config import get_settings
from app.llm.client import get_llm_client
from app.rag.embeddings import get_embedder
from app.rag.store import VectorStore

app = FastAPI(
    title="Equity Research Agent",
    description="Agentowy system analizy spółek giełdowych (LangGraph + tool-calling).",
    version="0.2.0",
)

# Budujemy agenta raz na proces. Zależności wybierane z .env; fabryki schodzą
# do atrap (brak klucza LLM / brak modelu embeddingów), więc aplikacja zawsze wstaje.
_settings = get_settings()
_agent = build_graph(
    llm=get_llm_client(_settings),
    store=VectorStore(_settings.database_url),
    embedder=get_embedder(_settings.embed_provider),
)


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"], description="Symbol spółki, np. AAPL.")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/analyze", response_model=Report)
def analyze(req: AnalyzeRequest) -> Report:
    """Uruchamia agenta na podanym tickerze i zwraca ustrukturyzowany raport."""
    result = _agent.invoke({"ticker": req.ticker})

    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])

    return result["report"]
