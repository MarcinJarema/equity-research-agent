"""Testy narzędzi.

Skupiamy się na metrics.py — to czyste funkcje, więc testujemy je bez sieci,
na ręcznie skonstruowanych danych. To pokazuje, czemu oddzieliliśmy pobieranie
danych (market_data) od liczenia (metrics).
"""

from datetime import date, timedelta

from app.tools.market_data import PricePoint
from app.tools.metrics import TRADING_DAYS_PER_MONTH, momentum_12_1, upside


def _history(prices: list[float]) -> list[PricePoint]:
    """Buduje listę punktów cenowych z listy cen (daty kolejne dni wstecz)."""
    start = date(2024, 1, 1)
    return [PricePoint(day=start + timedelta(days=i), close=p) for i, p in enumerate(prices)]


def test_momentum_none_when_history_too_short():
    # Mniej niż ~12 miesięcy danych => nie da się policzyć.
    assert momentum_12_1(_history([100.0] * 50)) is None


def test_momentum_positive_trend():
    # 13 miesięcy rosnących cen: cena 1mc temu > cena 12mc temu => momentum > 0.
    n = 13 * TRADING_DAYS_PER_MONTH
    prices = [100.0 + i for i in range(n)]  # rosnąco
    result = momentum_12_1(_history(prices))
    assert result is not None and result > 0


def test_momentum_exact_value():
    # Kontrolujemy dokładne punkty: 12mc temu = indeks -252, 1mc temu = indeks -21.
    n = 13 * TRADING_DAYS_PER_MONTH
    prices = [1.0] * n
    prices[-12 * TRADING_DAYS_PER_MONTH] = 100.0  # cena 12 mc temu
    prices[-1 * TRADING_DAYS_PER_MONTH] = 120.0   # cena 1 mc temu
    result = momentum_12_1(_history(prices))
    assert result == (120.0 / 100.0 - 1.0)  # +20%


def test_upside_basic():
    # target 120, cena 100 => +20%.
    assert upside(100.0, 120.0) == 0.2


def test_upside_none_without_target():
    assert upside(100.0, None) is None


def test_upside_none_when_price_zero():
    assert upside(0.0, 120.0) is None
