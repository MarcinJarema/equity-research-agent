"""Testy regułowego ratingu i scoringu rekomendacji (czyste funkcje, bez sieci)."""

from app.tools.rating import classify_rating
from app.tools.revisions import recommendation_score


def test_strong_buy_when_high_upside_and_momentum():
    r = classify_rating(upside=0.25, momentum_12_1=0.15, revisions_negative_2m=False)
    assert r == "Strong Buy"


def test_negative_revisions_demote_strong_buy_to_buy():
    # Te same mocne metryki, ale rewizje w dół od 2 miesięcy (-2 pkt) => degradacja.
    r = classify_rating(upside=0.25, momentum_12_1=0.15, revisions_negative_2m=True)
    assert r == "Buy"


def test_negative_revisions_can_flip_to_sell():
    # Umiarkowany upside (+1), zerowe momentum, rewizje negatywne (-2) => -1 => Sell.
    r = classify_rating(upside=0.06, momentum_12_1=0.0, revisions_negative_2m=True)
    assert r == "Sell"


def test_hold_when_neutral():
    r = classify_rating(upside=0.0, momentum_12_1=0.0, revisions_negative_2m=False)
    assert r == "Hold"


def test_sell_when_price_above_consensus_and_weak_momentum():
    r = classify_rating(upside=-0.10, momentum_12_1=-0.15, revisions_negative_2m=False)
    assert r == "Sell"


def test_recommendation_score_all_strong_buy_is_one():
    assert recommendation_score(10, 0, 0, 0, 0) == 1.0


def test_recommendation_score_mix():
    # 1 strongBuy(1) + 1 hold(3) => średnia 2.0
    assert recommendation_score(1, 0, 1, 0, 0) == 2.0


def test_recommendation_score_no_votes_is_none():
    assert recommendation_score(0, 0, 0, 0, 0) is None
