"""Behavioural tests for the pure Corpus -> Chunk transform (#11, spec #9).

The transform is a pure module, so these tests exercise it directly (no seam):
real corpus files offline for coverage, synthetic documents for precise
splitting behaviour. Chunk provision metadata is what Citations are later
derived from, so it must be exact.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.chunking import MAX_CHUNK_TOKENS, chunk_regulation, estimate_tokens
from src.models import Chunk, ProvisionKind

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "regulations"
CORPUS_FILES = {
    "gdpr": "gdpr.json",
    "ai-act": "ai-act.json",
    "dora": "dora.json",
}

EXPECTED_ARTICLE_COUNTS = {"gdpr": 99, "ai-act": 113, "dora": 64}
EXPECTED_RECITAL_COUNTS_WITH_TEXT = {"gdpr": 0, "ai-act": 0, "dora": 106}
EXPECTED_ANNEX_COUNTS = {"gdpr": 0, "ai-act": 13, "dora": 0}


def load_document(source_id: str) -> dict:
    return json.loads((DATA_DIR / CORPUS_FILES[source_id]).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def chunks_by_source() -> dict[str, list[Chunk]]:
    return {sid: chunk_regulation(load_document(sid)) for sid in CORPUS_FILES}


def all_chunks() -> list[Chunk]:
    return [c for chunks in chunks_by_source().values() for c in chunks]


def normalize(text: str) -> str:
    return " ".join(text.split())


def assert_text_preserved(source_text: str, joined_chunks: str) -> None:
    """Nothing lost: the whole text, else every line, else every sentence of an
    oversized line must appear contiguously somewhere in the provision's chunks.
    Adjacent fragments may legitimately land in different chunks once splitting
    kicks in, so contiguity is asserted per fragment, never across them."""
    cleaned = normalize(source_text)
    if not cleaned:
        return
    if cleaned in joined_chunks:
        return
    for line in source_text.splitlines():
        cleaned_line = normalize(line)
        if not cleaned_line or cleaned_line in joined_chunks:
            continue
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.;])\s+", cleaned_line)
            if sentence.strip()
        ]
        for sentence in sentences:
            assert sentence in joined_chunks, f"lost text around: {sentence[:80]!r}"


def source_articles(document: dict) -> list[dict]:
    """Chapter-nested articles, deduplicated by id (the corpus repeats two AI Act ids)."""
    seen = set()
    articles = []
    for chapter in document["chapters"]:
        for section in chapter.get("sections", []):
            for article in section.get("articles", []):
                if article["id"] not in seen:
                    seen.add(article["id"])
                    articles.append(article)
    return articles


# --- Corpus coverage: nothing dropped ---------------------------------------


@pytest.mark.parametrize("source_id", sorted(CORPUS_FILES))
def test_all_source_articles_are_chunked(source_id):
    document = load_document(source_id)
    expected = {a["number"] for a in source_articles(document)}
    chunked = {
        c.article_number for c in chunks_by_source()[source_id] if c.kind is ProvisionKind.article
    }
    assert len(expected) == EXPECTED_ARTICLE_COUNTS[source_id]
    assert chunked == expected


def test_ai_act_duplicate_articles_are_not_duplicated():
    # The lexplorer JSON lists articles 71 and 73 twice with identical content.
    for number in (71, 73):
        groups = [
            (c.chunk_index,)
            for c in chunks_by_source()["ai-act"]
            if c.kind is ProvisionKind.article and c.article_number == number and c.chunk_index == 0
        ]
        assert len(groups) == 1


@pytest.mark.parametrize("source_id", sorted(CORPUS_FILES))
def test_recitals_with_text_are_chunked(source_id):
    document = load_document(source_id)
    expected = {r["number"] for r in document["recitals"] if r.get("text", "").strip()}
    assert len(expected) == EXPECTED_RECITAL_COUNTS_WITH_TEXT[source_id]
    chunked = {
        c.recital_number for c in chunks_by_source()[source_id] if c.kind is ProvisionKind.recital
    }
    assert chunked == expected


@pytest.mark.parametrize("source_id", sorted(CORPUS_FILES))
def test_all_annexes_are_chunked(source_id):
    document = load_document(source_id)
    expected = {i for i, _ in enumerate(document["annexes"], start=1)}
    assert len(expected) == EXPECTED_ANNEX_COUNTS[source_id]
    chunked = {
        c.annex_number for c in chunks_by_source()[source_id] if c.kind is ProvisionKind.annex
    }
    assert chunked == expected


def test_dora_definitions_list_is_not_lost_content():
    # The transform chunks provisions only (Article/Recital/Annex), so DORA's
    # separate `definitions` list is skipped on the assumption its terms all
    # live inside Article 3's text. This tripwire pins that assumption.
    document = load_document("dora")
    joined = normalize(
        " ".join(
            c.text
            for c in chunks_by_source()["dora"]
            if c.kind is ProvisionKind.article and c.article_number == 3
        )
    )
    for definition in document["definitions"]:
        assert definition["term"] in joined, f"definition '{definition['term']}' left Article 3"


# --- Chunk shape: exactly-one-target invariant ------------------------------


def test_every_chunk_targets_exactly_one_provision_kind():
    for chunk in all_chunks():
        targets = [chunk.article_number, chunk.recital_number, chunk.annex_number]
        assert sum(t is not None for t in targets) == 1
        kind_field = {
            ProvisionKind.article: chunk.article_number,
            ProvisionKind.recital: chunk.recital_number,
            ProvisionKind.annex: chunk.annex_number,
        }[chunk.kind]
        assert kind_field is not None, "kind must agree with the populated number"


def test_chunk_rejects_multiple_provision_targets():
    with pytest.raises(ValidationError):
        Chunk(
            source_id="gdpr",
            kind=ProvisionKind.article,
            article_number=5,
            recital_number=1,
            text="Article 5",
        )


def test_chunk_rejects_missing_provision_target():
    with pytest.raises(ValidationError):
        Chunk(source_id="gdpr", kind=ProvisionKind.article, text="orphan text")


def test_chunk_rejects_kind_mismatch():
    with pytest.raises(ValidationError):
        Chunk(
            source_id="gdpr",
            kind=ProvisionKind.recital,
            article_number=5,
            text="Recital text",
        )


# --- Splitting behaviour against the real corpus ----------------------------


def test_long_article_splits_at_paragraph_boundaries():
    # DORA Article 28 has ten paragraphs totalling well above the budget.
    chunks = [
        c
        for c in chunks_by_source()["dora"]
        if c.kind is ProvisionKind.article and c.article_number == 28
    ]
    assert len(chunks) >= 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.num_chunks for c in chunks} == {len(chunks)}
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= MAX_CHUNK_TOKENS


def test_giant_definition_paragraph_splits_at_points():
    # GDPR Article 4 and DORA Article 3 hold dozens of definitions as points of
    # a single paragraph; splitting must go below paragraph level there.
    for source_id, number in (("gdpr", 4), ("dora", 3)):
        chunks = [
            c
            for c in chunks_by_source()[source_id]
            if c.kind is ProvisionKind.article and c.article_number == number
        ]
        assert len(chunks) >= 3, f"{source_id} article {number} should split"
        for chunk in chunks:
            assert estimate_tokens(chunk.text) <= MAX_CHUNK_TOKENS
        joined = normalize(" ".join(c.text for c in chunks))
        marker = "'processor' means" if source_id == "gdpr" else "'legacy ICT system' means"
        assert marker in joined


def test_unstructured_blob_stays_within_budget_via_sentence_fallback():
    # AI Act Article 3 is a single flat ~17k-character definitions blob with no
    # points in the JSON; sentence-level packing keeps every chunk near budget.
    chunks = [
        c
        for c in chunks_by_source()["ai-act"]
        if c.kind is ProvisionKind.article and c.article_number == 3
    ]
    assert len(chunks) >= 4
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= MAX_CHUNK_TOKENS
    joined = normalize(" ".join(c.text for c in chunks))
    assert "'AI system' means" in joined or "AI system" in joined


def test_recitals_are_never_split():
    # Recital integrity is locked: a recital is always exactly one chunk even
    # when it exceeds the token budget (only sanctioned oversize).
    for chunk in all_chunks():
        if chunk.kind is ProvisionKind.recital:
            assert chunk.num_chunks == 1


def test_token_budget_holds_outside_the_recital_exception():
    for chunk in all_chunks():
        if chunk.kind is not ProvisionKind.recital:
            assert estimate_tokens(chunk.text) <= MAX_CHUNK_TOKENS, (
                f"{chunk.source_id} {chunk.kind} {chunk.article_number or chunk.annex_number} "
                f"part {chunk.chunk_index} exceeds the budget"
            )


# --- No source text is lost --------------------------------------------------


@pytest.mark.parametrize("source_id", sorted(CORPUS_FILES))
def test_no_article_text_is_dropped(source_id):
    document = load_document(source_id)
    chunks = chunks_by_source()[source_id]
    for article in source_articles(document):
        joined = normalize(
            " ".join(
                c.text
                for c in chunks
                if c.kind is ProvisionKind.article and c.article_number == article["number"]
            )
        )
        for paragraph in article["paragraphs"]:
            assert_text_preserved(paragraph.get("text") or "", joined)
            for point in paragraph.get("points") or []:
                assert normalize(point.get("text") or "") in joined
                for sub_point in point.get("subPoints") or []:
                    assert normalize(sub_point.get("text") or "") in joined


def test_no_recital_or_annex_text_is_dropped():
    for source_id, document in ((sid, load_document(sid)) for sid in CORPUS_FILES):
        chunks = chunks_by_source()[source_id]
        for recital in document["recitals"]:
            if not recital.get("text", "").strip():
                continue
            joined = normalize(
                " ".join(
                    c.text
                    for c in chunks
                    if c.kind is ProvisionKind.recital and c.recital_number == recital["number"]
                )
            )
            assert normalize(recital["text"]) in joined
        for index, annex in enumerate(document["annexes"], start=1):
            joined = normalize(
                " ".join(c.text for c in chunks if c.kind is ProvisionKind.annex and c.annex_number == index)
            )
            assert_text_preserved(annex["content"], joined)


# --- Scale and determinism ----------------------------------------------------


def test_total_chunk_count_is_in_the_expected_range():
    # DAY_1 research expected ~500-700 chunks across the three regulations.
    total = len(all_chunks())
    assert 600 <= total <= 850, f"unexpected corpus scale: {total} chunks"


def test_transform_is_deterministic():
    document = load_document("dora")
    assert chunk_regulation(document) == chunk_regulation(document)


# --- Synthetic documents: precise packing behaviour --------------------------


def make_document(articles=(), recitals=(), annexes=(), source_id="test-doc"):
    return {
        "metadata": {"id": source_id, "shortName": "Test Doc"},
        "chapters": (
            [{"id": "chapter-1", "sections": [{"id": "chapter-1-s1", "articles": list(articles)}]}]
            if articles
            else []
        ),
        "recitals": list(recitals),
        "annexes": list(annexes),
    }


def make_paragraph(number, words, points=None):
    para = {"id": f"p-{number}", "number": number, "text": "word " * words, "points": points or []}
    return para


def make_article(number, title, paragraphs):
    return {
        "id": f"article-{number}",
        "number": number,
        "title": title,
        "paragraphs": paragraphs,
    }


def test_short_article_is_a_single_labelled_chunk():
    document = make_document(
        articles=[make_article(2, "Scope", [make_paragraph(1, 50)])],
    )
    chunks = chunk_regulation(document)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_id == "test-doc"
    assert chunk.kind is ProvisionKind.article
    assert chunk.article_number == 2
    assert chunk.title == "Scope"
    assert chunk.chunk_index == 0
    assert chunk.num_chunks == 1
    assert chunk.text.startswith("Article 2: Scope")
    assert normalize(chunk.text).endswith(normalize(make_paragraph(1, 50)["text"]).rstrip())


def test_paragraph_packing_respects_budget_while_preferring_whole_paragraphs():
    # Three ~190-token paragraphs: the packer fits two under the budget and
    # starts a new chunk rather than splitting a paragraph that fits alone.
    paragraphs = [make_paragraph(i, 150) for i in (1, 2, 3)]
    document = make_document(articles=[make_article(1, "T", paragraphs)])
    chunks = chunk_regulation(document)
    assert len(chunks) == 2
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert {c.num_chunks for c in chunks} == {2}
    assert all(estimate_tokens(c.text) <= MAX_CHUNK_TOKENS for c in chunks)
    assert "(part 1/2)" in chunks[0].text.splitlines()[0]
    assert "(part 2/2)" in chunks[1].text.splitlines()[0]


def test_oversized_paragraph_splits_at_point_boundaries():
    points = [
        {"id": "pt-a", "label": "a", "text": "word " * 120, "subPoints": []},
        {"id": "pt-b", "label": "b", "text": "word " * 120, "subPoints": []},
        {"id": "pt-c", "label": "c", "text": "word " * 120, "subPoints": []},
    ]
    paragraph = make_paragraph(None, 5, points=points)
    document = make_document(articles=[make_article(9, "Definitions", [paragraph])])
    chunks = chunk_regulation(document)
    assert len(chunks) >= 2
    assert {c.article_number for c in chunks} == {9}
    joined = normalize(" ".join(c.text for c in chunks))
    for label in ("a", "b", "c"):
        assert f"({label})" in joined
    assert all(estimate_tokens(c.text) <= MAX_CHUNK_TOKENS for c in chunks)


def test_oversized_unstructured_paragraph_splits_at_sentences():
    sentences = ["Sentence number %d ends here. " % i for i in range(1, 60)]
    paragraph = {"id": "p-1", "number": 1, "text": "".join(sentences), "points": []}
    document = make_document(articles=[make_article(3, "Blob", [paragraph])])
    chunks = chunk_regulation(document)
    assert len(chunks) >= 2
    assert all(estimate_tokens(c.text) <= MAX_CHUNK_TOKENS for c in chunks)
    joined = normalize(" ".join(c.text for c in chunks))
    assert "Sentence number 1 ends here." in joined
    assert "Sentence number 59 ends here." in joined


def test_long_recital_stays_whole_in_one_chunk():
    document = make_document(
        recitals=[{"id": "recital-1", "number": 1, "text": "word " * 2000}],
    )
    chunks = chunk_regulation(document)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.kind is ProvisionKind.recital
    assert chunk.recital_number == 1
    assert chunk.num_chunks == 1
    assert chunk.text.startswith("Recital 1")
    assert estimate_tokens(chunk.text) > MAX_CHUNK_TOKENS  # sanctioned oversize


def test_annex_lines_pack_into_budgeted_chunks_under_one_heading():
    lines = "\n".join(f"Item %d of the annex list." % i for i in range(1, 80))
    document = make_document(
        annexes=[{"id": "annex-1", "title": "List", "content": lines}],
    )
    chunks = chunk_regulation(document)
    assert len(chunks) >= 2
    assert {c.annex_number for c in chunks} == {1}
    assert all(estimate_tokens(c.text) <= MAX_CHUNK_TOKENS for c in chunks)
    assert "(part 1/" in chunks[0].text.splitlines()[0]
    joined = normalize(" ".join(c.text for c in chunks))
    assert "Item 79 of the annex list." in joined


def test_identical_duplicate_articles_are_skipped():
    duplicated = make_article(5, "Same", [make_paragraph(1, 20)])
    document = make_document(articles=[duplicated, dict(duplicated)])
    chunks = chunk_regulation(document)
    assert len(chunks) == 1


def test_conflicting_duplicate_articles_fail_loudly():
    first = make_article(5, "Same", [make_paragraph(1, 20)])
    second = make_article(5, "Different", [make_paragraph(1, 21)])
    document = make_document(articles=[first, second])
    with pytest.raises(ValueError, match="article 5"):
        chunk_regulation(document)


def test_empty_article_fails_loudly():
    document = make_document(
        articles=[make_article(7, "Empty", [{"id": "p-1", "number": 1, "text": "", "points": []}])],
    )
    with pytest.raises(ValueError, match="article 7"):
        chunk_regulation(document)


def test_estimate_tokens_is_positive_and_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("a " * 100) > estimate_tokens("a " * 10)
