"""Testy agenta bez sieci.

Węzeł market_data woła yfinance (sieć), więc tu testujemy resztę na ręcznie
zbudowanym stanie i atrapie LLM — deterministycznie i za darmo.
"""

from datetime import date

from app.agent.nodes import make_synthesize_node, metrics_node, router_node
from app.agent.state import Report
from app.llm.client import FakeLLMClient
from app.tools.market_data import MarketData, PricePoint
from app.tools.metrics import Metrics


def _market_data() -> MarketData:
    return MarketData(
        ticker="TEST",
        currency="USD",
        current_price=100.0,
        volume=1000,
        target_mean_price=120.0,
        history=[PricePoint(day=date(2024, 1, 1), close=100.0)],
    )


def test_router_sets_plan():
    out = router_node({"ticker": "TEST"})
    assert out["plan"] == ["market_data", "metrics"]


def test_metrics_node_computes_from_market_data():
    out = metrics_node({"market_data": _market_data()})
    assert isinstance(out["metrics"], Metrics)
    assert out["metrics"].upside == 0.2  # (120-100)/100


def test_synthesize_builds_report_with_our_metrics():
    node = make_synthesize_node(FakeLLMClient())
    metrics = Metrics(momentum_12_1=0.1, upside=0.2)
    out = node({"market_data": _market_data(), "metrics": metrics})

    report = out["report"]
    assert isinstance(report, Report)
    # Metryki w raporcie to NASZE liczby, nie wymysł LLM.
    assert report.metrics.upside == 0.2
    assert report.disclaimer  # disclaimer zawsze obecny
    assert report.recommendation  # coś zwrócił (atrapa: "Neutralna")


def test_synthesize_skips_on_error():
    node = make_synthesize_node(FakeLLMClient())
    out = node({"error": "boom"})
    assert out == {}  # przy błędzie nie syntetyzujemy
