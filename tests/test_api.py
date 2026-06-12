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
