"""Testy modułu ewaluacji (logika odporności, bez sieci)."""

from eval.run_eval import _as_int


def test_as_int_parses_valid():
    assert _as_int(4) == 4
    assert _as_int("5") == 5


def test_as_int_falls_back_on_garbage():
    # Sędzia (LLM) może zwrócić "4/5", "N/A" albo None — nie wywalamy runu.
    assert _as_int("4/5") == 0
    assert _as_int("N/A") == 0
    assert _as_int(None) == 0
    assert _as_int(2.9) == 2  # int() obcina, ale nie rzuca
