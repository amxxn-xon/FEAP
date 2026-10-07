"""Unit tests for engine.py. Run with: python -m pytest"""
from datetime import date

import pytest

import engine
from engine import (
    AIWordlistExpander, ContextDecoder, CorpusProcessor, DiachronicTimeline,
    EntitySpiderweb, SyntacticXRay, ThreatAssessor, count_hedges,
    count_logistic_verbs, estimate_email_date, extract_dates, find_hedges,
    keyword_pattern, normalize_entity,
)

CORPUS = [
    {"id": "e1", "sender": "A", "subject": "Lunch",
     "body": "Could you bring a pizza to the house? I think Maria would like that."},
    {"id": "e2", "sender": "B", "subject": "Dessert",
     "body": "Pizzas were delivered. Please arrange the car for Paris on May 5."},
    {"id": "e3", "sender": "C", "subject": "Piece",
     "body": "A piece of pie. The mayor sent the package."},
    {"id": "e4", "sender": "D", "subject": "Hot dogs",
     "body": "Two hot   dogs, please."},
]


# ── Keyword matching ─────────────────────────────────────────────────────────
def test_whole_word_match_excludes_substrings():
    pat = keyword_pattern("pie")
    assert pat.search("a slice of pie")
    assert not pat.search("a piece of cake")


def test_whole_word_match_allows_inflections_and_case():
    pat = keyword_pattern("pizza")
    assert pat.search("Pizzas were delivered")
    assert pat.search("PIZZA!")
    assert keyword_pattern("model").search("she was modeling")


def test_substring_mode_keeps_old_behaviour():
    assert keyword_pattern("pie", whole_word=False).search("a piece")


def test_multiword_keyword_and_special_chars():
    assert keyword_pattern("hot dog").search("two hot   dogs")
    assert keyword_pattern("c++").search("I like c++ a lot")
    assert not keyword_pattern("   ").search("anything")


def test_query_by_keyword_whole_word():
    hits = CorpusProcessor.query_by_keyword(CORPUS, "pie")
    assert [e["id"] for e in hits] == ["e3"]
    hits = CorpusProcessor.query_by_keyword(CORPUS, "pie", whole_word=False)
    assert [e["id"] for e in hits] == ["e3"]  # "piece" and "pie" are both in e3
    assert [e["id"] for e in CorpusProcessor.query_by_keyword(CORPUS, "pizza")] == ["e1", "e2"]


def test_extract_sentences_with_keyword():
    rows = CorpusProcessor.extract_sentences_with_keyword(CORPUS[:2], "pizza")
    assert [r["sentence"] for r in rows] == [
        "Could you bring a pizza to the house?", "Pizzas were delivered."]


# ── Hedges & logistic verbs ──────────────────────────────────────────────────
def test_hedges_are_word_boundary_aware():
    assert find_hedges("The mayor said maybe") == ["maybe"]
    assert count_hedges("I think it could work") == 2      # multi-word + modal
    assert count_hedges("I THINK so") == 1                 # case-insensitive


def test_may_as_month_is_not_a_hedge():
    assert count_hedges("Meet on May 5 or in May.") == 0
    assert count_hedges("May I ask? You may.") == 2


def test_logistic_verbs_match_inflections():
    assert count_logistic_verbs("I arranged it, sent it and am picking up Bob") == 3
    assert count_logistic_verbs("The arrangement was nice") == 0


# ── Dates ────────────────────────────────────────────────────────────────────
def test_extract_dates_formats():
    text = ("On Apr 6, 2016, at 3:04 PM wrote. Date: Thu, 07 Jan 2016. "
            "Fri Nov 12 14:40:17 2010. 2009/5/4. July 2. 2013. 3/30/2018. Feb 30, 2012")
    assert set(extract_dates(text)) == {
        date(2016, 4, 6), date(2016, 1, 7), date(2010, 11, 12),
        date(2009, 5, 4), date(2013, 7, 2), date(2018, 3, 30)}


def test_estimate_email_date_uses_latest():
    assert estimate_email_date("Hi. On Jan 2, 2012 Bob wrote: see Dec 1, 2011") == date(2012, 1, 2)
    assert estimate_email_date("no dates here") is None


# ── Entities ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,clean", [
    ("the Plaza Hotel's", "Plaza Hotel"),
    ("  Paris , ", "Paris"),
    ("New\nYork", "New York"),
])
def test_normalize_entity(raw, clean):
    assert normalize_entity(raw) == clean


# ── Corpus loading ───────────────────────────────────────────────────────────
def test_real_corpus_loads_and_ids_resolve():
    clean = CorpusProcessor.load_clean()
    annotated = CorpusProcessor.load_annotated()
    assert len(clean) == len(annotated) > 0
    assert "pos_tagged" not in annotated[0]
    first = clean[0]
    assert CorpusProcessor.get_email_by_id(first["id"]) is first
    assert CorpusProcessor.get_email_by_id("does-not-exist") is None


# ── Modules ──────────────────────────────────────────────────────────────────
def test_timeline_corpus_order_and_densities():
    ann = [dict(e, flag_count=i) for i, e in enumerate(CORPUS)]
    out = DiachronicTimeline().analyse("pizza", annotated_corpus=ann, order="corpus")
    assert out["order"] == "corpus"
    assert [r["email_id"] for r in out["timeline"]] == ["e1", "e2"]
    assert out["timeline"][0]["hedge_density"] > 0


def test_timeline_falls_back_without_dates():
    out = DiachronicTimeline().analyse("pizza", annotated_corpus=CORPUS, order="date")
    assert out["order"] == "corpus"
    assert DiachronicTimeline().analyse("zzz", annotated_corpus=CORPUS) == {
        "error": "No emails found for keyword 'zzz'."}


def test_timeline_real_corpus_is_chronological():
    out = DiachronicTimeline().analyse("massage")
    assert out["order"] == "date"
    dates = [r["date"] for r in out["timeline"]]
    assert dates == sorted(dates)




def test_lda_topic_ids_map_to_matching_emails():
    pytest.importorskip("gensim")
    out = ContextDecoder().analyse("pizza", num_topics=3, interpret_topics=False)
    hit_ids = {e["id"] for e in CorpusProcessor.query_by_keyword(CorpusProcessor.load_clean(), "pizza")}
    assert out["doc_topics"]
    assert all(d["email_id"] in hit_ids for d in out["doc_topics"])
    for topic in out["topics"]:
        assert "pizza" not in topic["words"]          # keyword excluded
        assert not {"wrote", "iphone", "sent"} & set(topic["words"])  # boilerplate excluded


def test_lda_too_few_emails():
    pytest.importorskip("gensim")
    out = ContextDecoder().analyse("pizza", corpus=CORPUS, interpret_topics=False)
    assert "error" in out


def _spacy_ready():
    try:
        engine.get_spacy_nlp()
        return True
    except RuntimeError:
        return False


spacy_required = pytest.mark.skipif(not _spacy_ready(), reason="spaCy model not installed")


@spacy_required
def test_spiderweb_edges_only_reference_returned_nodes():
    out = EntitySpiderweb().analyse("pizza", max_emails=60, max_nodes=25)
    ids = {n["id"] for n in out["nodes"]}
    assert len(ids) <= 25
    assert all(e["source"] in ids and e["target"] in ids for e in out["edges"])
    assert set(out["node_email_map"]) == ids
    assert not ids & {"Warmly", "Thanks"}


@spacy_required
def test_xray_flags_keyword_tokens_and_hedges():
    out = SyntacticXRay().analyse("pizza", corpus=CORPUS)
    assert out["sentences_analysed"] == 2
    first = out["sentence_details"][0]
    assert [t["text"] for t in first["tokens"] if t["is_keyword"]] == ["pizza"]
    assert out["hedge_percentage"] == 50.0       # "Could …" vs "Pizzas were delivered."


# ── Zero-shot pieces (model replaced by a stub) ──────────────────────────────
class _FakePipe:
    """Mimics transformers' zero-shot pipeline output shape."""
    def __call__(self, texts, candidate_labels, multi_label=False, batch_size=None):
        def one(text):
            scores = [0.7] + [0.3 / (len(candidate_labels) - 1)] * (len(candidate_labels) - 1)
            return {"sequence": text, "labels": list(candidate_labels), "scores": scores}
        return one(texts) if isinstance(texts, str) else [one(t) for t in texts]


@pytest.fixture
def fake_pipeline(monkeypatch):
    monkeypatch.setattr(engine, "_ZS_PIPELINE", _FakePipe())
    monkeypatch.setattr(AIWordlistExpander, "_cache", {})


def test_threat_assessor_batches_and_reports_progress(fake_pipeline):
    seen = []
    out = ThreatAssessor().assess_many(["a", "b", "c"], ["X", "Y"], batch_size=2,
                                       progress=lambda d, t: seen.append((d, t)))
    assert [r["top_label"] for r in out] == ["X", "X", "X"]
    assert out[0]["top_score"] == 70.0
    assert seen == [(2, 3), (3, 3)]


def test_threat_assessor_requires_two_labels(fake_pipeline):
    with pytest.raises(ValueError):
        ThreatAssessor().assess_email("text", ["Only one", "Only one "])


def test_wordlist_expander_uses_batched_scores(fake_pipeline):
    expanded = AIWordlistExpander.expand_corpus_wordlist(["lunch", "pizza"], "food_code")
    assert {"lunch", "pizza"} <= expanded


def test_wordlist_expander_falls_back_to_static(monkeypatch):
    monkeypatch.setattr(AIWordlistExpander, "_cache", {})

    def boom(*a, **k):
        raise RuntimeError("no model")
    monkeypatch.setattr(engine, "get_zero_shot_pipeline", boom)
    assert AIWordlistExpander.is_member("pizza", "food_code") is True
    assert AIWordlistExpander.is_member("lunch", "food_code") is False
