"""Narzędzie calc_metrics: liczy metryki na podstawie danych rynkowych.

Dwie metryki w MVP:
  • momentum 12-1  — sygnał "pędu" ceny,
  • upside         — potencjał wzrostu vs konsensus analityków.

Funkcje są czyste (przyjmują liczby/listę punktów), więc testujemy je
deterministycznie, bez odpytywania yfinance.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.tools.market_data import MarketData, PricePoint

# Ile dni handlowych przypada średnio na miesiąc (do okien momentum).
TRADING_DAYS_PER_MONTH = 21


class Metrics(BaseModel):
    """Policzone metryki dla spółki. None = brak danych do policzenia."""

    momentum_12_1: float | None = None  # zwrot w ułamku, np. 0.18 = +18%
    upside: float | None = None         # (target - cena) / cena, ułamek


def momentum_12_1(history: list[PricePoint]) -> float | None:
    """Momentum 12-1: zwrot ceny od ~12 do ~1 miesiąca wstecz.

    DLACZEGO akurat "12 minus 1", a nie zwykły zwrot 12-miesięczny:
    w badaniach momentum pomija się NAJNOWSZY miesiąc, bo w krótkim terminie
    występuje efekt odwrócenia (short-term reversal) — świeży ruch często się
    cofa i zaszumiłby sygnał. Dlatego mierzymy okno t-12mc → t-1mc.

    Wzór: (cena 1 mc temu / cena 12 mc temu) - 1.
    Zwraca None, gdy historia jest za krótka.
    """
    needed = 12 * TRADING_DAYS_PER_MONTH  # ~252 dni handlowych wstecz
    if len(history) <= needed:
        return None

    price_1m_ago = history[-1 * TRADING_DAYS_PER_MONTH].close      # ~21 dni wstecz
    price_12m_ago = history[-12 * TRADING_DAYS_PER_MONTH].close    # ~252 dni wstecz
    if price_12m_ago == 0:
        return None

    return price_1m_ago / price_12m_ago - 1.0


def upside(current_price: float, target_mean_price: float | None) -> float | None:
    """Upside vs konsensus: (cena docelowa - cena bieżąca) / cena bieżąca.

    Cena docelowa to średnia z cen docelowych analityków (targetMeanPrice).
    Dodatni upside => analitycy widzą przestrzeń do wzrostu. Zwraca None,
    gdy brak konsensusu lub cena bieżąca = 0.
    """
    if target_mean_price is None or current_price == 0:
        return None
    return (target_mean_price - current_price) / current_price


def calc_metrics(data: MarketData) -> Metrics:
    """Składa wszystkie metryki z surowych danych rynkowych."""
    return Metrics(
        momentum_12_1=momentum_12_1(data.history),
        upside=upside(data.current_price, data.target_mean_price),
    )
