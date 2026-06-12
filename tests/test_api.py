"""Testy warstwy API — walidacja wejścia (bez sieci)."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_rejects_invalid_ticker():
    # Znaki spoza dozwolonego zbioru => 422 z walidacji, zanim ruszy jakakolwiek sieć.
    r = client.post("/analyze", json={"ticker": "AAPL; DROP TABLE documents;--"})
    assert r.status_code == 422


def test_analyze_rejects_empty_ticker():
    r = client.post("/analyze", json={"ticker": "   "})
    assert r.status_code == 422


def test_analyze_maps_unexpected_error_to_502(monkeypatch):
    # Nieoczekiwany wyjątek w agencie (np. awaria yfinance/LLM) -> czytelny 502,
    # nie 500 ze stack trace. Podmieniamy invoke, żeby nie ruszać sieci.
    import app.main as main

    def boom(*_args, **_kwargs):
        raise RuntimeError("symulowana awaria")

    monkeypatch.setattr(main._agent, "invoke", boom)
    r = client.post("/analyze", json={"ticker": "AAPL"})
    assert r.status_code == 502
