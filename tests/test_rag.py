"""Testy RAG.

Chunking testujemy jako czystą logikę (bez sieci/bazy). FakeEmbedder pozwala
sprawdzić determinizm i wymiar wektora bez pobierania modelu (torch).
"""

from app.rag.embeddings import EMBED_DIM, FakeEmbedder
from app.rag.ingest import chunk_text, split_sentences


def test_split_sentences():
    sents = split_sentences("Spółka rośnie. Wyniki dobre! Czy utrzyma tempo?")
    assert sents == ["Spółka rośnie.", "Wyniki dobre!", "Czy utrzyma tempo?"]


def test_chunk_text_groups_sentences_under_target():
    # Krótki tekst => jeden chunk.
    chunks = chunk_text("Zdanie jedno. Zdanie dwa.", target_chars=500)
    assert len(chunks) == 1


def test_chunk_text_splits_long_text_with_overlap():
    # Dużo długich zdań => kilka chunków; nakładka => zdanie powtarza się na styku.
    sentences = [f"To jest dosc dlugie zdanie numer {i} z trescia." for i in range(10)]
    text = " ".join(sentences)
    chunks = chunk_text(text, target_chars=80)
    assert len(chunks) > 1
    # Nakładka: ostatnie zdanie chunku N pojawia się na początku chunku N+1.
    assert chunks[0].split(". ")[-1].strip(". ") in chunks[1]


def test_fake_embedder_is_deterministic_and_right_dim():
    emb = FakeEmbedder()
    a = emb.embed(["tekst"])[0]
    b = emb.embed(["tekst"])[0]
    assert a == b                # determinizm
    assert len(a) == EMBED_DIM   # wymiar zgodny ze schematem tabeli
