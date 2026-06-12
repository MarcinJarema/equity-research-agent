# Equity Research Agent

Agentowy system analizy spółek giełdowych oparty o LLM. Dla podanego **tickera**
pobiera dane rynkowe, liczy metryki (momentum, upside vs konsensus analityków),
pobiera newsy, wyszukuje kontekst w bazie wektorowej (RAG) i przez LLM składa
**ustrukturyzowany raport analityczny**.

> ⚠️ **Disclaimer:** To jest **narzędzie analityczne i projekt edukacyjny**, a
> **nie porada inwestycyjna**. Wyniki mogą zawierać błędy i halucynacje modelu.
> Nie podejmuj decyzji inwestycyjnych na podstawie tego narzędzia.

## Stack

| Warstwa | Technologia |
|---|---|
| Orkiestracja agenta | LangGraph (graf: router → tools → synthesize) |
| API | FastAPI (`POST /analyze`) |
| LLM | abstrakcja `LLMClient` — domyślnie Groq/Llama, wymienne na OpenAI/Anthropic |
| RAG | pgvector (rozszerzenie PostgreSQL) |
| Dane rynkowe | yfinance |
| Observability | LangSmith (trace) |
| Ewaluacja | moduł `eval/` z LLM-as-judge |
| Konteneryzacja | Docker + docker-compose |

## Architektura

```
Użytkownik (ticker spółki)
      │
      ▼
  FastAPI /analyze
      │
      ▼
  LangGraph Agent (orkiestrator)
      ├── tool: get_market_data   (ceny, wolumen — yfinance)
      ├── tool: calc_metrics      (momentum, upside vs konsensus)
      ├── tool: get_news          (ostatnie newsy)
      ├── tool: rag_search        (kontekst z bazy wektorowej)
      └── node: synthesize        (LLM składa raport + ocenę)
      │
      ▼
  Strukturalny raport (JSON + tekst) + trace w LangSmith
```

## Status

🚧 W budowie. Projekt powstaje etapami:

- [ ] **ETAP 1** — szkielet FastAPI + Docker + narzędzia `market_data` i `metrics`
- [ ] **ETAP 2** — agent LangGraph + abstrakcja LLM + structured output
- [ ] **ETAP 3** — RAG (ingest + chunking + pgvector) jako narzędzie agenta
- [ ] **ETAP 4** — LangSmith + ewaluacja (LLM-as-judge) + finalne README

## Uruchomienie

> Instrukcja docelowa (docker-compose) pojawi się w ETAPIE 1.

```bash
cp .env.example .env   # uzupełnij klucze
# docker compose up --build   # wkrótce
```
