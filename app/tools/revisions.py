"""Narzędzie get_revision_trend: trend rekomendacji analityków (yfinance).

Sygnał, którego szukamy: czy konsensus analityków POGORSZYŁ się przez ostatnie
~2 miesiące (rewizje w dół). yfinance.recommendations zwraca rozkład rekomendacji
w oknach czasowych (0m, -1m, -2m, -3m) z kolumnami strongBuy/buy/hold/sell/strongSell.

Liczymy średnią ważoną w skali 1..5 (1 = Strong Buy, 5 = Strong Sell — im NIŻEJ
tym bardziej byczo). Jeśli wynik dziś > wynik sprzed 2 miesięcy => konsensus stał
się bardziej niedźwiedzi => rewizje negatywne od 2 miesięcy.
"""

from __future__ import annotations

import logging

import yfinance as yf
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Wagi w skali analitycznej: 1=Strong Buy ... 5=Strong Sell (niżej = lepiej).
_WEIGHTS = {"strongBuy": 1, "buy": 2, "hold": 3, "sell": 4, "strongSell": 5}


class RevisionTrend(BaseModel):
    """Trend rekomendacji. negative_2m = konsensus pogorszył się przez 2 miesiące."""

    available: bool = False
    score_now: float | None = None       # 1..5, niżej = bardziej byczo
    score_2m_ago: float | None = None
    negative_2m: bool = False


def recommendation_score(
    strong_buy: float, buy: float, hold: float, sell: float, strong_sell: float
) -> float | None:
    """Średnia ważona rekomendacji w skali 1..5. None, gdy brak głosów."""
    counts = {
        "strongBuy": strong_buy, "buy": buy, "hold": hold,
        "sell": sell, "strongSell": strong_sell,
    }
    total = sum(counts.values())
    if total <= 0:
        return None
    weighted = sum(_WEIGHTS[k] * v for k, v in counts.items())
    return weighted / total


def get_revision_trend(ticker: str) -> RevisionTrend:
    """Pobiera trend rekomendacji i wykrywa pogorszenie konsensusu w 2 miesiące."""
    ticker = ticker.strip().upper()
    try:
        rec = yf.Ticker(ticker).recommendations
    except Exception as exc:
        logger.warning("get_revision_trend(%s) nieudane: %s", ticker, exc)
        return RevisionTrend()

    if rec is None or getattr(rec, "empty", True):
        return RevisionTrend()

    # 'period' bywa kolumną — ustawiamy ją jako indeks, by sięgać po '0m' / '-2m'.
    df = rec.set_index("period") if "period" in rec.columns else rec

    def score_for(period: str) -> float | None:
        if period not in df.index:
            return None
        row = df.loc[period]
        return recommendation_score(
            float(row.get("strongBuy", 0) or 0),
            float(row.get("buy", 0) or 0),
            float(row.get("hold", 0) or 0),
            float(row.get("sell", 0) or 0),
            float(row.get("strongSell", 0) or 0),
        )

    now = score_for("0m")
    two_ago = score_for("-2m")
    negative = now is not None and two_ago is not None and now > two_ago + 1e-9
    return RevisionTrend(
        available=now is not None,
        score_now=now,
        score_2m_ago=two_ago,
        negative_2m=negative,
    )
