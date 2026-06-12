"""Narzędzie get_news: ostatnie newsy spółki z yfinance.

Świadomie NIE używamy przeglądarki/scrapingu — yfinance oddaje newsy przez API,
co jest prostsze i stabilniejsze. Struktura .news bywa różna między wersjami
yfinance, więc parsujemy ją obronnie.
"""

from __future__ import annotations

import logging

import yfinance as yf
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class NewsItem(BaseModel):
    title: str
    summary: str = ""
    publisher: str = ""


def get_news(ticker: str, *, limit: int = 8) -> list[NewsItem]:
    """Pobiera do `limit` najnowszych newsów. Przy błędzie zwraca pustą listę."""
    ticker = ticker.strip().upper()
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as exc:
        logger.warning("get_news(%s) nieudane, zwracam pustą listę: %s", ticker, exc)
        return []

    items: list[NewsItem] = []
    for entry in raw[:limit]:
        item = _parse_entry(entry)
        if item and item.title:
            items.append(item)
    return items


def _parse_entry(entry: dict) -> NewsItem | None:
    """Wyciąga tytuł/streszczenie/wydawcę z różnych kształtów wpisu yfinance."""
    # Nowsze yfinance pakuje treść w entry['content'].
    content = entry.get("content") if isinstance(entry.get("content"), dict) else entry

    title = content.get("title") or entry.get("title") or ""
    summary = content.get("summary") or content.get("description") or entry.get("summary") or ""

    publisher = ""
    provider = content.get("provider") or entry.get("publisher")
    if isinstance(provider, dict):
        publisher = provider.get("displayName", "")
    elif isinstance(provider, str):
        publisher = provider

    if not title:
        return None
    return NewsItem(title=str(title), summary=str(summary), publisher=str(publisher))
