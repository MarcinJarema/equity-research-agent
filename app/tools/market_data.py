"""Narzędzie get_market_data: pobiera dane rynkowe spółki z yfinance.

W ETAPIE 1 to zwykła funkcja wołana wprost z endpointu /analyze.
W ETAPIE 2 opakujemy ją jako "tool" agenta LangGraph (ten sam rdzeń, inna obudowa).

Świadomie oddzielamy POBIERANIE danych (tu) od LICZENIA metryk (metrics.py) —
dzięki temu metryki testujemy bez sieci, na zwstrzykniętych liczbach.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf
from pydantic import BaseModel


class PricePoint(BaseModel):
    """Pojedynczy punkt na osi czasu — data + cena zamknięcia."""

    day: date
    close: float


class MarketData(BaseModel):
    """Surowe dane rynkowe dla jednego tickera (wejście do liczenia metryk)."""

    ticker: str
    currency: str | None = None
    current_price: float
    volume: int | None = None
    # Cena docelowa z konsensusu analityków (do policzenia upside). Bywa niedostępna.
    target_mean_price: float | None = None
    # Historia dziennych zamknięć — potrzebna do momentum (min. ~13 miesięcy).
    history: list[PricePoint]


class MarketDataError(Exception):
    """Rzucany, gdy nie da się pobrać sensownych danych dla tickera."""


def get_market_data(ticker: str, *, period: str = "14mo") -> MarketData:
    """Pobiera dane rynkowe dla tickera.

    Args:
        ticker: symbol spółki, np. "AAPL".
        period: ile historii pobrać. Domyślnie 14 miesięcy — momentum 12-1
            potrzebuje ceny sprzed ~12 i ~1 miesiąca, bierzemy zapas na luki/weekendy.

    Raises:
        MarketDataError: gdy ticker nie zwraca żadnej historii cen.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise MarketDataError("Pusty ticker.")

    tk = yf.Ticker(ticker)

    # history() zwraca DataFrame z kolumną 'Close'. auto_adjust=True => ceny
    # skorygowane o splity/dywidendy, co jest poprawne do liczenia momentum.
    hist = tk.history(period=period, auto_adjust=True)
    # Odrzucamy wiersze bez ceny zamknięcia (niedomknięty bieżący dzień bywa NaN) —
    # NaN zepsułby cenę bieżącą i metryki, a w JSON to nieprawidłowy token.
    hist = hist.dropna(subset=["Close"])
    if hist.empty:
        raise MarketDataError(f"Brak danych historycznych dla tickera '{ticker}'.")

    history = [
        PricePoint(day=idx.date(), close=float(row["Close"]))
        for idx, row in hist.iterrows()
    ]

    # .info bywa wolne i czasem niekompletne — czytamy ostrożnie, z fallbackami.
    info = _safe_info(tk)
    current_price = float(history[-1].close)  # ostatnie zamknięcie jako bieżąca cena
    last_volume = hist["Volume"].iloc[-1]

    return MarketData(
        ticker=ticker,
        currency=info.get("currency"),
        current_price=current_price,
        # pd.notna: NaN -> None (brak danych), ale wolumen 0 traktujemy jako prawdziwe 0.
        volume=int(last_volume) if pd.notna(last_volume) else None,
        target_mean_price=_as_float(info.get("targetMeanPrice")),
        history=history,
    )


def _safe_info(tk: yf.Ticker) -> dict:
    """yfinance .info potrafi rzucić wyjątkiem — izolujemy to."""
    try:
        return tk.info or {}
    except Exception:
        return {}


def _as_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
