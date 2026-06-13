"""Regułowa klasyfikacja: Strong Buy / Buy / Hold / Sell.

To DETERMINISTYCZNY scoring (nie LLM) — przejrzysty i tani, idealny do porównania
wielu spółek naraz. Łączy trzy sygnały:
  • upside vs konsensus  (potencjał wzrostu),
  • momentum 12-1        (pęd ceny),
  • rewizje 2-miesięczne (pogorszenie konsensusu analityków => mocna kara).

Negatywne rewizje od 2 miesięcy świadomie WAŻĄ dużo (-2 pkt): nawet przy dodatnim
upside potrafią zdegradować ocenę, bo oznaczają, że analitycy psują prognozy.
"""

from __future__ import annotations

# Kolejność od najlepszej do najgorszej (przydaje się do sortowania w UI).
RATINGS = ["Strong Buy", "Buy", "Hold", "Sell"]


def classify_rating(
    upside: float | None,
    momentum_12_1: float | None,
    revisions_negative_2m: bool,
) -> str:
    """Mapuje sygnały na rating. Patrz docstring modułu po uzasadnienie wag."""
    score = 0

    if upside is not None:
        if upside > 0.20:
            score += 2          # duży potencjał wzrostu
        elif upside > 0.05:
            score += 1
        elif upside < -0.05:
            score -= 1          # cena powyżej konsensusu => downside

    if momentum_12_1 is not None:
        if momentum_12_1 > 0.10:
            score += 1
        elif momentum_12_1 < -0.10:
            score -= 1

    if revisions_negative_2m:
        score -= 2              # rewizje w dół od 2 miesięcy = mocny sygnał ostrzegawczy

    if score >= 3:
        return "Strong Buy"
    if score >= 1:
        return "Buy"
    if score >= 0:
        return "Hold"
    return "Sell"
