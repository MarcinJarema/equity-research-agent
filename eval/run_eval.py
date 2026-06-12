"""Ewaluacja jakości raportów agenta: LLM-as-judge + sprawdzenia deterministyczne.

DLACZEGO hybryda (LLM + kod), a nie sam LLM:
  • disclaimer i obecność metryk sprawdzamy KODEM — to fakty binarne, LLM tu
    tylko by szumiał i potrafi się mylić;
  • trafność, kompletność i halucynację liczb ocenia LLM-judge — to wymaga
    rozumienia treści, nie da się tego sprowadzić do if-a.

Sędzia dostaje POLICZONE metryki jako prawdę — dzięki temu może wykryć, czy
raport nie zmyślił liczb spoza danych wejściowych.

Uruchomienie:
    python -m eval.run_eval                 # cały dataset
    python -m eval.run_eval AAPL MSFT       # wybrane tickery
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.agent.graph import build_graph
from app.agent.state import Report
from app.core.config import get_settings
from app.llm.client import LLMClient, get_llm_client
from app.rag.embeddings import get_embedder
from app.rag.store import VectorStore

DATASET = Path(__file__).parent / "dataset.jsonl"

JUDGE_SYSTEM = (
    "Jesteś surowym recenzentem raportów giełdowych. Oceniasz raport względem "
    "METRYK referencyjnych (to jedyne prawdziwe liczby). Zwróć WYŁĄCZNIE JSON: "
    "{\"relevance\": 1-5, \"completeness\": 1-5, \"no_number_hallucination\": true/false, "
    "\"comment\": \"krótko\"}. "
    "relevance: czy ocena wynika z metryk i kontekstu. completeness: czy są "
    "rekomendacja, uzasadnienie i ryzyka. no_number_hallucination: false, jeśli "
    "raport podaje konkretne liczby, których NIE ma w metrykach referencyjnych."
)


def load_cases(filter_tickers: list[str]) -> list[dict]:
    cases = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if filter_tickers:
        wanted = {t.upper() for t in filter_tickers}
        cases = [c for c in cases if c["ticker"].upper() in wanted]
    return cases


def deterministic_checks(report: Report) -> dict:
    """Fakty binarne sprawdzane kodem (pewniejsze niż pytanie LLM)."""
    return {
        "has_disclaimer": bool(report.disclaimer.strip()),
        "has_recommendation": bool(report.recommendation.strip()),
        "has_metrics": report.metrics is not None,
    }


def judge(llm: LLMClient, report: Report) -> dict:
    """LLM-as-judge: ocena jakościowa raportu względem metryk referencyjnych."""
    payload = {
        "reference_metrics": {
            "momentum_12_1": report.metrics.momentum_12_1,
            "upside": report.metrics.upside,
        },
        "report": {
            "recommendation": report.recommendation,
            "rationale": report.rationale,
            "risks": report.risks,
        },
    }
    raw = llm.complete_json(JUDGE_SYSTEM, json.dumps(payload, ensure_ascii=False))
    return {
        "relevance": int(raw.get("relevance", 0)),
        "completeness": int(raw.get("completeness", 0)),
        "no_number_hallucination": bool(raw.get("no_number_hallucination", False)),
        "comment": str(raw.get("comment", "")),
    }


def main() -> None:
    settings = get_settings()
    llm = get_llm_client(settings)
    agent = build_graph(
        llm=llm,
        store=VectorStore(settings.database_url),
        embedder=get_embedder(settings.embed_provider),
    )

    cases = load_cases(sys.argv[1:])
    judge_id = f"{settings.llm_provider}/{settings.llm_model}"
    print(f"Ewaluacja {len(cases)} przypadków (sędzia: {judge_id})\n")

    scores: list[float] = []
    for case in cases:
        ticker = case["ticker"]
        result = agent.invoke({"ticker": ticker})

        if result.get("error") or "report" not in result:
            print(f"  {ticker:6} | BŁĄD: {result.get('error', 'brak raportu')}")
            scores.append(0.0)
            continue

        report: Report = result["report"]
        det = deterministic_checks(report)
        jud = judge(llm, report)

        # Wynik łączny: średnia ocen LLM (skala 0-1) z karami za braki deterministyczne.
        quality = (jud["relevance"] + jud["completeness"]) / 10.0  # 0..1
        if not jud["no_number_hallucination"]:
            quality *= 0.5  # halucynacja liczb to poważny błąd
        if not all(det.values()):
            quality *= 0.5  # brak disclaimera/metryk/rekomendacji
        scores.append(quality)

        flags = "".join(
            "✓" if v else "✗" for v in (det["has_disclaimer"], jud["no_number_hallucination"])
        )
        print(
            f"  {ticker:6} | jakość={quality:.2f} | rel={jud['relevance']} "
            f"comp={jud['completeness']} | disc/no-halu={flags} | {jud['comment'][:60]}"
        )

    if scores:
        print(f"\nŚrednia jakość: {sum(scores) / len(scores):.2f} (0-1), n={len(scores)}")


if __name__ == "__main__":
    main()
