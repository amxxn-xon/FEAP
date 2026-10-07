"""
engine.py — Forensic Entity-Action Profiler
Core OOP engine: CorpusProcessor base class + 5 analytical subclasses.

Lemalock Hypothesis — Computational Pragmatics Framework
Author: Ameen K.P | EFLU

Design notes:
- Heavy models (spaCy, the zero-shot transformer) are loaded once per process
  and shared by every module.
- Keyword matching is whole-word by default (with simple inflections), so
  "pie" no longer matches "piece" and "may" no longer matches "mayor".
- Every module degrades gracefully when its optional dependency is missing.
"""

import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

# ── Optional heavy imports (graceful fallback) ───────────────────────────────
try:
    import spacy  # type: ignore
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    import networkx as nx  # type: ignore
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

try:
    import pandas as pd  # type: ignore
    _PD_AVAILABLE = True
except ImportError:
    _PD_AVAILABLE = False

try:
    from gensim import corpora  # type: ignore
    from gensim.models import LdaModel  # type: ignore
    from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOP  # type: ignore
    _GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_STOP = frozenset()
    _GENSIM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_CLEAN_PATH      = os.path.join(_BASE, "data", "epstein_clean_data.json")
_ANNOTATED_PATH  = os.path.join(_BASE, "data", "epstein_annotated.json")

SPACY_MODEL     = "en_core_web_sm"
ZERO_SHOT_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

# ─────────────────────────────────────────────────────────────────────────────
# STATIC WORDLISTS (seeds — the AIWordlistExpander can enrich them at runtime)
# ─────────────────────────────────────────────────────────────────────────────
FOOD_CODE_WORDS = {
    'pizza', 'hotdog', 'hot dog', 'pasta', 'ice cream', 'cheese',
    'burger', 'walnut', 'sauce', 'chicken', 'menu', 'dessert',
    'appetizer', 'entree', 'recipe', 'catering', 'buffet',
    'cookie', 'candy', 'cake', 'pie', 'donut', 'muffin',
    'beef jerky', 'jerky', 'milk', 'grape juice', 'grape'
}

EUPHEMISM_WORDS = {
    'massage', 'party', 'entertainment', 'model', 'modeling',
    'young', 'girl', 'girls', 'friend', 'friends', 'special',
    'arrangement', 'gift', 'package', 'delivery', 'service',
    'visit', 'visitor', 'guest', 'travel', 'trip', 'island',
    'private', 'discreet', 'favor', 'introduce', 'fresh'
}

HEDGE_WORDS = {
    'would', 'could', 'might', 'should', 'may', 'perhaps', 'possibly',
    'maybe', 'likely', 'probably', 'apparently', 'seemingly', 'sort of',
    'kind of', 'i think', 'i believe', 'i suppose', 'presumably'
}

LOGISTIC_VERBS = {
    'schedule', 'transport', 'arrange', 'coordinate', 'deliver', 'send',
    'move', 'transfer', 'book', 'reserve', 'confirm', 'organize',
    'prepare', 'handle', 'manage', 'pick up', 'drop off', 'bring', 'provide'
}

# Words that carry no topical signal in this corpus: e-mail headers, client
# boilerplate, dates and redaction markers. Used by LDA and word-frequency
# fallbacks so topics describe content rather than e-mail plumbing.
EMAIL_BOILERPLATE = {
    'wrote', 'sent', 'iphone', 'ipad', 'blackberry', 'forwarded', 'forward',
    'fwd', 'message', 'original', 'redacted', 'efta', 'subject', 'date',
    'from', 'cc', 'bcc', 'reply', 'mailto', 'com', 'net', 'org', 'www',
    'http', 'https', 'html', 'gmail', 'mail', 'email', 'emails', 'image',
    'jpg', 'png', 'pdf', 'attachment', 'begin', 'edt', 'est', 'pdt', 'pst',
    'gmt', 'cdt', 'utc', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday',
    'sunday', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept',
    'oct', 'nov', 'dec', 'january', 'february', 'march', 'april', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'thanks', 'thank', 'regards', 'best', 'dear', 'hello', 'hi', 'please',
    'jeffrey', 'epstein', 'jeff', 'information', 'contained', 'intended',
    'recipient', 'confidential', 'privileged', 'unsubscribe',
}

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

# ─────────────────────────────────────────────────────────────────────────────
# TEXT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
# Simple inflectional endings accepted in whole-word mode:
# model → models / modeling / modelled, massage → massages / massaged.
_INFLECTION_SUFFIX = r"(?:s|es|d|ed|ing|er|ers|'s|’s)?"


def keyword_pattern(keyword: str, whole_word: bool = True) -> "re.Pattern[str]":
    """
    Compile a case-insensitive pattern for `keyword`.

    whole_word=True  → match the word (plus simple inflections) only, so
                       "pie" does not match "piece".
    whole_word=False → plain substring match (the original behaviour).
    Multi-word keywords tolerate any run of whitespace between words.
    """
    parts = keyword.strip().split()
    if not parts:
        # Never-matching pattern for empty input.
        return re.compile(r"(?!x)x")
    body = r"\s+".join(re.escape(p) for p in parts)
    if whole_word:
        body = rf"(?<!\w){body}{_INFLECTION_SUFFIX}(?!\w)"
    return re.compile(body, re.IGNORECASE)


def _alternation(phrases: Iterable[str]) -> str:
    # Longest first so multi-word phrases win over their prefixes.
    escaped = sorted((r"\s+".join(map(re.escape, p.split())) for p in phrases),
                     key=len, reverse=True)
    return "|".join(escaped)


_HEDGE_RE = re.compile(rf"(?<!\w)(?:{_alternation(HEDGE_WORDS)})(?!\w)", re.IGNORECASE)
# "May" as a month: "May 5", "5 May", "in May", "of May 2014"…
_MAY_MONTH_BEFORE = re.compile(r"(?:\d|\b(?:in|of|early|late|mid|since|until|by))[\s,-]*$", re.IGNORECASE)
_MAY_MONTH_AFTER  = re.compile(r"^[\s,]*\d")


def find_hedges(text: str) -> List[str]:
    """Return every modal hedge in `text` (word-boundary aware, case-insensitive)."""
    found = []
    for m in _HEDGE_RE.finditer(text):
        word = m.group(0)
        if word == "May" and (
            _MAY_MONTH_AFTER.match(text[m.end():m.end() + 6])
            or _MAY_MONTH_BEFORE.search(text[max(0, m.start() - 12):m.start()])
        ):
            continue
        found.append(word.lower())
    return found


def count_hedges(text: str) -> int:
    return len(find_hedges(text))


def has_hedge(text: str) -> bool:
    return bool(find_hedges(text))


# Irregular / doubled-consonant forms that the regular rules below miss.
_IRREGULAR_FORMS = {
    "send":     ["sent"],
    "bring":    ["brought"],
    "drop off": ["dropped off", "dropping off"],
    "pick up":  ["picked up", "picking up", "picks up"],
}


def _verb_forms(verb: str) -> List[str]:
    head, *rest = verb.split()
    tail = (" " + " ".join(rest)) if rest else ""
    if head.endswith("e"):
        stem = head[:-1]
        forms = [head, head + "s", head + "d", stem + "ing"]
    else:
        forms = [head, head + "s", head + "ed", head + "ing"]
    forms = [f + tail for f in forms]
    return forms + _IRREGULAR_FORMS.get(verb, [])


_LOGISTIC_RE = re.compile(
    rf"(?<!\w)(?:{_alternation(f for v in LOGISTIC_VERBS for f in _verb_forms(v))})(?!\w)",
    re.IGNORECASE,
)


def count_logistic_verbs(text: str) -> int:
    """Count logistic verbs in any regular/irregular inflection (arranged, sent…)."""
    return len(_LOGISTIC_RE.findall(text))


def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


# ── Date recovery ────────────────────────────────────────────────────────────
# The corpus has no date field, but forwarded headers and reply lines carry
# dates ("On Apr 6, 2016, at 3:04 PM … wrote", "Date: Thu, 07 Jan 2016").
_MON = (r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
        r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_YEAR = r"((?:19|20)\d\d)"
_DATE_PATTERNS = [
    # Apr 6, 2016 · July 2. 2013 · September 8th 2013
    (re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?[.,]?\s+{_YEAR}\b"), ("m", "d", "y")),
    # 07 Jan 2016 · 7 January, 2016
    (re.compile(rf"\b(\d{{1,2}})\s+{_MON}\.?,?\s+{_YEAR}\b"), ("d", "m", "y")),
    # Fri Nov 12 14:40:17 2010
    (re.compile(rf"\b{_MON}\s+(\d{{1,2}})\s+\d{{1,2}}:\d{{2}}(?::\d{{2}})?\s+{_YEAR}\b"), ("m", "d", "y")),
    # 2009/5/4 · 2009-05-04
    (re.compile(rf"\b{_YEAR}[/-](\d{{1,2}})[/-](\d{{1,2}})\b"), ("y", "m", "d")),
    # 3/30/2018 (US order)
    (re.compile(rf"\b(\d{{1,2}})/(\d{{1,2}})/{_YEAR}\b"), ("m", "d", "y")),
]
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def extract_dates(text: str) -> List[date]:
    """Every plausible calendar date mentioned in `text`."""
    found = []
    for pattern, order in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            parts = dict(zip(order, m.groups()))
            month = parts["m"]
            month = _MONTHS[month[:3].lower()] if not month.isdigit() else int(month)
            try:
                d = date(int(parts["y"]), month, int(parts["d"]))
            except ValueError:
                continue
            if 1990 <= d.year <= date.today().year:
                found.append(d)
    return found


def estimate_email_date(text: str) -> Optional[date]:
    """
    Best-effort send date: the most recent date mentioned. Quoted replies and
    forwarded headers are older than the message that contains them.
    """
    dates = extract_dates(text)
    return max(dates) if dates else None


# ─────────────────────────────────────────────────────────────────────────────
# SHARED MODEL LOADERS (one instance per process, thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_LOCK = threading.Lock()
_NLP = None
_ZS_PIPELINE = None
# (message, timestamp) of the last failed pipeline load. Loading can hang on
# network retries when the model hub is unreachable, so failures are
# remembered briefly instead of being retried on every call.
_ZS_ERROR: Optional[tuple] = None
_ZS_RETRY_AFTER = 300  # seconds


def get_spacy_nlp():
    """Load the spaCy English model once and share it across modules."""
    global _NLP
    if _NLP is None:
        if not _SPACY_AVAILABLE:
            raise RuntimeError(
                "spaCy is not installed. Run: pip install spacy && "
                f"python -m spacy download {SPACY_MODEL}"
            )
        with _MODEL_LOCK:
            if _NLP is None:
                try:
                    _NLP = spacy.load(SPACY_MODEL)
                except OSError:
                    raise RuntimeError(
                        f"spaCy model '{SPACY_MODEL}' is missing. Run: "
                        f"python -m spacy download {SPACY_MODEL}"
                    )
    return _NLP


def get_zero_shot_pipeline():
    """
    Load the zero-shot NLI pipeline once and share it across the threat
    assessor, topic interpreter and wordlist expander (previously each loaded
    its own copy of the same model).
    """
    global _ZS_PIPELINE, _ZS_ERROR
    if _ZS_PIPELINE is None:
        with _MODEL_LOCK:
            if _ZS_PIPELINE is None:
                if _ZS_ERROR and time.monotonic() - _ZS_ERROR[1] < _ZS_RETRY_AFTER:
                    raise RuntimeError(_ZS_ERROR[0])
                try:
                    from transformers import pipeline  # type: ignore
                    _ZS_PIPELINE = pipeline(
                        "zero-shot-classification",
                        model=ZERO_SHOT_MODEL,
                        device=-1,
                    )
                except ImportError:
                    msg = ("transformers or torch not installed. "
                           "Run: pip install transformers torch")
                    _ZS_ERROR = (msg, time.monotonic())
                    raise RuntimeError(msg)
                except Exception as e:
                    msg = f"Error initializing NLP pipeline: {e}"
                    _ZS_ERROR = (msg, time.monotonic())
                    raise RuntimeError(msg)
                _ZS_ERROR = None
    return _ZS_PIPELINE


def _zero_shot(texts: Sequence[str], labels: Sequence[str],
               batch_size: int = 8) -> List[dict]:
    """Run the shared pipeline on a batch and always return a list of results."""
    pipe = get_zero_shot_pipeline()
    out = pipe(list(texts), candidate_labels=list(labels),
               multi_label=False, batch_size=batch_size)
    return [out] if isinstance(out, dict) else list(out)


# ─────────────────────────────────────────────────────────────────────────────
# AI WORDLIST EXPANDER
# Uses the zero-shot pipeline to score & expand word membership in forensic
# categories on top of the static seed dictionaries.
# ─────────────────────────────────────────────────────────────────────────────
class AIWordlistExpander:
    """
    Uses a zero-shot NLI model to decide whether frequent corpus words belong
    to forensic categories (food cipher, euphemism, logistical verb, hedge).
    Falls back to the static dictionaries when the model is unavailable.
    """
    # Cache: "word::category" → bool
    _cache: Dict[str, bool] = {}

    CATEGORIES = {
        "food_code": "This word is used as a food-related code word or culinary euphemism in illicit communications",
        "euphemism": "This word is used as a euphemism or coded language to conceal illicit activity",
        "logistic":  "This word describes a logistical coordination action like scheduling, transporting, or arranging",
        "hedge":     "This word is a modal hedge that reduces commitment or obscures intent in written communication",
    }
    _NEUTRAL = "This word has no special significance"

    STATIC_BASES = {
        "food_code": FOOD_CODE_WORDS,
        "euphemism": EUPHEMISM_WORDS,
        "logistic":  LOGISTIC_VERBS,
        "hedge":     HEDGE_WORDS,
    }

    @classmethod
    def _static_fallback(cls, word: str, category: str) -> bool:
        return word.lower() in cls.STATIC_BASES.get(category, set())

    @classmethod
    def score_words(cls, words: Sequence[str], category: str,
                    threshold: float = 0.65) -> Dict[str, bool]:
        """Classify many words for one category in a single batched call."""
        todo = [w for w in words if f"{w}::{category}" not in cls._cache]
        if todo:
            label_text = cls.CATEGORIES.get(category, category)
            try:
                outs = _zero_shot(
                    [f'The word "{w}" appears in an email.' for w in todo],
                    [label_text, cls._NEUTRAL],
                )
                for w, out in zip(todo, outs):
                    score = dict(zip(out["labels"], out["scores"])).get(label_text, 0.0)
                    cls._cache[f"{w}::{category}"] = score >= threshold
            except Exception:
                for w in todo:
                    cls._cache[f"{w}::{category}"] = cls._static_fallback(w, category)
        return {w: cls._cache[f"{w}::{category}"] for w in words}

    @classmethod
    def is_member(cls, word: str, category: str, threshold: float = 0.65) -> bool:
        return cls.score_words([word], category, threshold)[word]

    @classmethod
    def expand_corpus_wordlist(cls, corpus_words: List[str], category: str,
                               top_n: int = 40) -> set:
        """
        Scan the most frequent corpus words and return those the AI classifies
        as belonging to `category`, merged with the static seed set.
        """
        base = set(cls.STATIC_BASES.get(category, set()))
        candidates = [w for w in corpus_words[:top_n] if w not in base]
        scored = cls.score_words(candidates, category)
        return base | {w for w, ok in scored.items() if ok}

    @classmethod
    def get_dynamic_wordlists(cls, corpus, top_n: int = 40) -> dict:
        """
        Build all four dynamic wordlists from corpus vocabulary.
        Returns dict with keys: food_code, euphemism, logistic, hedge.
        """
        stop = set(GENSIM_STOP) | EMAIL_BOILERPLATE
        counts: Counter = Counter()
        for e in corpus[:500]:
            counts.update(t for t in tokenize_words(e.get("body", ""))
                          if len(t) >= 3 and t not in stop)
        top_words = [w for w, _ in counts.most_common(100)]
        return {cat: cls.expand_corpus_wordlist(top_words, cat, top_n=top_n)
                for cat in cls.CATEGORIES}

# ─────────────────────────────────────────────────────────────────────────────
# AI TOPIC INTERPRETER
# Matches LDA topic word clusters to forensic semantic categories.
# ─────────────────────────────────────────────────────────────────────────────
class AITopicInterpreter:
    """
    Uses zero-shot classification to match LDA topic word clusters to
    a set of forensic semantic categories, producing interpretable labels
    and descriptions.
    """

    FORENSIC_TOPIC_LABELS = [
        "Logistical coordination and travel planning",
        "Social grooming and relationship management",
        "Financial transactions and payments",
        "Coded food/service euphemisms for illicit activity",
        "Legal and administrative communication",
        "Interpersonal pressure and coercion",
        "Media and public relations management",
        "General routine correspondence",
        "Event planning and scheduling",
        "Identity concealment and information suppression",
    ]

    _NOTES = {
        "Logistical coordination and travel planning":
            "🚨 High forensic relevance — coordinates movement of people/assets.",
        "Social grooming and relationship management":
            "🔴 Critical — language patterns consistent with grooming behaviour.",
        "Financial transactions and payments":
            "🔶 Significant — may indicate payment for services rendered covertly.",
        "Coded food/service euphemisms for illicit activity":
            "🟡 Cipher alert — food/service language as coded substitution.",
        "Legal and administrative communication":
            "🔵 Monitor — may contain suppression or legal-cover strategies.",
        "Interpersonal pressure and coercion":
            "🔴 Critical — language consistent with coercive control.",
        "Media and public relations management":
            "🟠 Notable — possible reputation management / narrative control.",
        "General routine correspondence":
            "⚪ Low forensic weight — appears to be standard communication.",
        "Event planning and scheduling":
            "🟡 Contextual — scheduling language may encode real logistics.",
        "Identity concealment and information suppression":
            "🔴 Critical — signals deliberate information control.",
    }

    @classmethod
    def interpret_topic(cls, words, weights=None) -> dict:
        """
        Given LDA topic words, return:
          - best_label: matched forensic category
          - confidence: score 0–100
          - description: plain-English explanation (plain text)
          - forensic_significance: brief forensic note
        """
        if not words:
            return cls._fallback_interpretation(words)
        text = f"An email cluster whose most distinctive words are: {', '.join(words[:6])}."
        try:
            result = _zero_shot([text], cls.FORENSIC_TOPIC_LABELS)[0]
        except Exception:
            return cls._fallback_interpretation(words)

        best_label = result["labels"][0]
        confidence = round(result["scores"][0] * 100, 1)
        return {
            "best_label": best_label,
            "confidence": confidence,
            "description": cls._build_description(best_label, words, confidence),
            "forensic_significance": cls._forensic_note(best_label),
            "ai_powered": True,
        }

    @classmethod
    def _build_description(cls, label: str, words, confidence: float) -> str:
        top = ", ".join(f'"{w}"' for w in words[:4])
        conf_text = "strongly" if confidence > 70 else "moderately" if confidence > 45 else "tentatively"
        return (
            f"This topic cluster {conf_text} matches the pattern of {label}. "
            f"Key signal words include {top}. AI confidence: {confidence}%."
        )

    @classmethod
    def _forensic_note(cls, label: str) -> str:
        return cls._NOTES.get(label, "ℹ️ Review manually for contextual significance.")

    @classmethod
    def _fallback_interpretation(cls, words) -> dict:
        """Rule-based fallback when transformer is unavailable."""
        food_signals     = {'food', 'pizza', 'chicken', 'dinner', 'lunch', 'menu', 'cake', 'jerky', 'pasta'}
        logistic_signals = {'schedule', 'transport', 'arrange', 'book', 'travel', 'flight', 'trip', 'island'}
        coerce_signals   = {'must', 'need', 'expect', 'required', 'demand', 'force'}
        money_signals    = {'payment', 'paid', 'pay', 'money', 'wire', 'invoice', 'bank', 'account'}

        top = set(words[:8])
        if top & food_signals:
            label = "Coded food/service euphemisms for illicit activity"
        elif top & logistic_signals:
            label = "Logistical coordination and travel planning"
        elif top & money_signals:
            label = "Financial transactions and payments"
        elif top & coerce_signals:
            label = "Interpersonal pressure and coercion"
        else:
            label = "General routine correspondence"

        return {
            "best_label": label,
            "confidence": 0,
            "description": f"Rule-based classification (AI model unavailable). Key words: {', '.join(words[:4])}.",
            "forensic_significance": "⚠️ AI model unavailable — classification is heuristic only.",
            "ai_powered": False,
        }

# ─────────────────────────────────────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────
class CorpusProcessor:
    """Base class: loads & caches the corpus, provides shared utilities."""
    _clean_cache = None
    _annotated_cache: Dict[bool, list] = {}
    _id_index: Optional[Dict[str, dict]] = None

    # ── Loading ──────────────────────────────────────────────────────────────
    @classmethod
    def load_clean(cls):
        if cls._clean_cache is None:
            with open(_CLEAN_PATH, "r", encoding="utf-8") as f:
                cls._clean_cache = json.load(f)
        return cls._clean_cache

    @classmethod
    def load_annotated(cls, include_pos: bool = False):
        """
        Load the annotated corpus. The per-token `pos_tagged` field makes up
        most of the 15 MB file and nothing in the dashboard reads it, so it is
        dropped unless include_pos=True.
        """
        if include_pos not in cls._annotated_cache:
            with open(_ANNOTATED_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not include_pos:
                data = [{k: v for k, v in e.items() if k != "pos_tagged"} for e in data]
            cls._annotated_cache[include_pos] = data
        return cls._annotated_cache[include_pos]

    @classmethod
    def get_email_by_id(cls, email_id: str):
        """O(1) lookup of a full email record by its ID."""
        if cls._id_index is None:
            cls._id_index = {e.get("id"): e for e in cls.load_clean()}
        return cls._id_index.get(email_id)

    # ── Querying ─────────────────────────────────────────────────────────────
    @staticmethod
    def query_by_keyword(corpus, keyword: str, whole_word: bool = True) -> List[Dict[str, Any]]:
        pat = keyword_pattern(keyword, whole_word)
        return [e for e in corpus
                if pat.search(e.get("body", "")) or pat.search(e.get("subject", ""))]

    @staticmethod
    def extract_sentences_with_keyword(emails, keyword: str,
                                       whole_word: bool = True) -> List[Dict[str, Any]]:
        pat = keyword_pattern(keyword, whole_word)
        result = []
        for e in emails:
            for s in re.split(r'(?<=[.!?])\s+', e.get("body", "")):
                if pat.search(s):
                    result.append({
                        "id": e["id"], "sender": e.get("sender", ""),
                        "subject": e.get("subject", ""), "sentence": s.strip(),
                    })
        return result

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[█]+', '[REDACTED]', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def corpus_stats(corpus) -> dict:
        total_tokens = sum(len(e.get("body", "").split()) for e in corpus)
        senders = Counter(e.get("sender", "Unknown") for e in corpus)
        return {
            "total_emails": len(corpus),
            "total_tokens": total_tokens,
            "unique_senders": len(senders),
            "top_senders": senders.most_common(10),
        }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — SYNTACTIC X-RAY  (spaCy dependency parsing)
# ─────────────────────────────────────────────────────────────────────────────
class SyntacticXRay(CorpusProcessor):
    """Dependency-parse sentences containing the target keyword."""

    def _get_nlp(self):
        return get_spacy_nlp()

    @staticmethod
    def _root_action(root) -> str:
        """Root lemma, including a phrasal particle when present ("pick up")."""
        prt = next((c for c in root.children if c.dep_ == "prt"), None)
        return f"{root.lemma_.lower()} {prt.text.lower()}" if prt is not None else root.lemma_.lower()

    def analyse(self, keyword: str, max_sentences: int = 30, corpus=None,
                whole_word: bool = True) -> dict:
        if corpus is None:
            corpus = self.load_clean()
        hits      = self.query_by_keyword(corpus, keyword, whole_word)
        sentences = self.extract_sentences_with_keyword(hits, keyword, whole_word)[:max_sentences]
        pat       = keyword_pattern(keyword, whole_word)

        results = []
        root_verbs: Counter = Counter()
        modifiers: Counter = Counter()
        hedge_count = 0
        logistic_count = 0

        nlp  = self._get_nlp()
        docs = nlp.pipe([item["sentence"] for item in sentences], batch_size=50)

        for item, doc in zip(sentences, docs):
            # Keyword tokens = tokens overlapping a keyword match (handles
            # multi-word keywords and inflections such as "models").
            spans = [(m.start(), m.end()) for m in pat.finditer(doc.text)]
            kw_idx = {t.i for t in doc
                      if any(t.idx < end and t.idx + len(t.text) > start for start, end in spans)}
            if not kw_idx:
                continue

            root = next((t for t in doc if t.dep_ == "ROOT"), None)
            is_logistic = False
            if root is not None:
                action = self._root_action(root)
                root_verbs[action] += 1
                is_logistic = action in LOGISTIC_VERBS or root.lemma_.lower() in LOGISTIC_VERBS
                logistic_count += is_logistic

            for i in kw_idx:
                for child in doc[i].children:
                    if child.i not in kw_idx and child.text.strip() and not (child.is_punct or child.is_stop):
                        modifiers[child.text.lower()] += 1

            hedges = find_hedges(item["sentence"])
            hedge_count += bool(hedges)

            results.append({
                "sentence": item["sentence"],
                "email_id": item["id"],
                "sender":   item["sender"],
                "subject":  item["subject"],
                "edges": [
                    {"from": t.head.text, "to": t.text, "dep": t.dep_}
                    for t in doc if t.dep_ != "punct"
                ],
                "tokens": [
                    {"text": t.text, "pos": t.pos_, "dep": t.dep_,
                     "head": t.head.text, "is_keyword": t.i in kw_idx}
                    for t in doc
                ],
                "root":        root.text if root is not None else "",
                "hedges":      hedges,
                "is_logistic": is_logistic,
            })

        parsed = max(len(results), 1)
        return {
            "keyword":                   keyword,
            "total_emails_with_keyword": len(hits),
            "sentences_analysed":        len(results),
            "hedge_percentage":          round(hedge_count / parsed * 100, 1),
            "logistic_verb_percentage":  round(logistic_count / parsed * 100, 1),
            "top_root_verbs":            root_verbs.most_common(10),
            "top_modifiers":             modifiers.most_common(10),
            "sentence_details":          results,
        }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — ENTITY SPIDERWEB  (spaCy NER + NetworkX)
# ─────────────────────────────────────────────────────────────────────────────
_ENTITY_LABELS = {"PERSON": "PERSON", "ORG": "ORGANIZATION", "GPE": "GPE",
                  "FAC": "FACILITY", "LOC": "LOCATION"}


def normalize_entity(name: str) -> str:
    """Tidy an entity mention so variants collapse onto one node."""
    name = re.sub(r"\s+", " ", name).strip(" \t\"'`.,;:()[]{}<>-–—")
    name = re.sub(r"(?:'s|’s)$", "", name)
    name = re.sub(r"^(?:the|The)\s+", "", name)
    return name.strip()


# Greetings / sign-offs that spaCy regularly mistakes for names.
_NON_ENTITIES = {
    "warmly", "thanks", "thank you", "regards", "best", "best regards", "cheers",
    "sincerely", "love", "xoxo", "hi", "hello", "dear", "sent", "iphone", "ipad",
    "fwd", "re", "subject", "cc", "bcc", "tel", "fax", "mobile", "email",
}


def _is_valid_entity(name: str) -> bool:
    return (len(name) > 2
            and any(c.isalpha() for c in name)
            and name.lower() not in _NON_ENTITIES
            and "redacted" not in name.lower()
            and not re.fullmatch(r"EFTA\w*", name))


class EntitySpiderweb(CorpusProcessor):
    """Named-entity co-occurrence graph centred on a keyword."""

    def _get_nlp(self):
        return get_spacy_nlp()

    def analyse(self, keyword: str, max_emails: int = 150, corpus=None,
                whole_word: bool = True, max_nodes: int = 60,
                max_edges: int = 100) -> dict:
        """
        Build the co-occurrence network. Only the `max_nodes` best-connected
        entities are returned and every edge references a returned node.
        `node_email_map` lets the UI list the emails behind a clicked node.
        """
        if corpus is None:
            corpus = self.load_clean()
        hits = self.query_by_keyword(corpus, keyword, whole_word)[:max_emails]

        edge_weights: Counter = Counter()
        type_votes: Dict[str, Counter] = defaultdict(Counter)
        mentions: Counter = Counter()
        node_email_ids: Dict[str, List[str]] = defaultdict(list)
        email_entity_map = []

        nlp  = self._get_nlp()
        docs = nlp.pipe((email.get("body", "")[:2000] for email in hits), batch_size=50)

        for email, doc in zip(hits, docs):
            names = set()
            for ent in doc.ents:
                label = _ENTITY_LABELS.get(ent.label_)
                if not label:
                    continue
                name = normalize_entity(ent.text)
                if not _is_valid_entity(name):
                    continue
                names.add(name)
                type_votes[name][label] += 1

            for name in names:
                mentions[name] += 1
                node_email_ids[name].append(email["id"])

            ordered = sorted(names)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    edge_weights[(ordered[i], ordered[j])] += 1

            if ordered:
                email_entity_map.append({"id": email["id"], "entities": ordered})

        # Weighted degree over the full graph decides which nodes are kept.
        strength: Counter = Counter()
        for (a, b), w in edge_weights.items():
            strength[a] += w
            strength[b] += w
        ranked = sorted(mentions, key=lambda n: (strength[n], mentions[n], n), reverse=True)
        keep = set(ranked[:max_nodes])

        edges = [
            {"source": a, "target": b, "weight": w}
            for (a, b), w in edge_weights.most_common()
            if a in keep and b in keep
        ][:max_edges]

        betweenness: Dict[str, float] = {}
        if _NX_AVAILABLE and edges:
            g = nx.Graph()
            for e in edges:
                # Strong ties = short distances for path-based centrality.
                g.add_edge(e["source"], e["target"], weight=e["weight"], distance=1.0 / e["weight"])
            betweenness = nx.betweenness_centrality(g, weight="distance", normalized=True)

        nodes = [
            {
                "id":          name,
                "type":        type_votes[name].most_common(1)[0][0],
                "mentions":    mentions[name],
                "degree":      strength[name],
                "betweenness": round(betweenness.get(name, 0.0), 4),
            }
            for name in ranked[:max_nodes]
        ]

        by_id = {e["id"]: e for e in hits}
        node_email_map = {
            name: [
                {k: by_id[eid].get(k, "") for k in ("id", "sender", "subject", "body")}
                for eid in node_email_ids[name]
            ]
            for name in keep
        }

        node_strength = Counter()
        for e in edges:
            node_strength[e["source"]] += e["weight"]
            node_strength[e["target"]] += e["weight"]

        return {
            "keyword":          keyword,
            "total_emails":     len(hits),
            "total_entities":   len(mentions),
            "nodes":            nodes,
            "edges":            edges,
            "node_email_map":   node_email_map,
            "email_entity_map": email_entity_map,
            "top_entities":     node_strength.most_common(15),
            "top_brokers":      sorted(
                ((n, b) for n, b in betweenness.items() if b > 0),
                key=lambda x: x[1], reverse=True)[:10],
        }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — HIDDEN CONTEXT DECODER  (Gensim LDA + AI topic interpretation)
# ─────────────────────────────────────────────────────────────────────────────
class ContextDecoder(CorpusProcessor):
    """
    Unsupervised LDA topic modeling on emails filtered by keyword, optionally
    enriched with topic descriptions via AITopicInterpreter.
    """
    _STOP = set(GENSIM_STOP) | EMAIL_BOILERPLATE | {
        'would', 'could', 'should', 'may', 'might', 'will', 'can', 'get', 'got',
        'let', 'know', 'think', 'want', 'like', 'well', 'one', 'time', 'see',
        'just', 'also', 'good', 'great', 'need', 'today', 'tomorrow', 'yes',
        'okay', 'sure', 'going', 'really', 'make', 'said', 'new',
    }

    def _tokenize(self, text: str, extra_stop: Iterable[str] = ()) -> list:
        stop = self._STOP | set(extra_stop)
        return [t for t in tokenize_words(text)
                if len(t) >= 3 and "'" not in t and t not in stop]

    @staticmethod
    def _keyword_stopwords(keyword: str) -> set:
        """The keyword itself and its inflections would dominate every topic."""
        stop = set()
        for w in tokenize_words(keyword):
            stop |= {w, w + "s", w + "es", w + "d", w + "ed", w + "ing", w + "er", w + "ers"}
        return stop

    def analyse(self, keyword: str, num_topics: int = 5,
                interpret_topics: bool = True, corpus=None,
                whole_word: bool = True) -> dict:
        if not _GENSIM_AVAILABLE:
            raise RuntimeError("Gensim is not installed. Run: pip install gensim")

        if corpus is None:
            corpus = self.load_clean()
        hits = self.query_by_keyword(corpus, keyword, whole_word)

        if len(hits) < 5:
            return {"error": f"Too few emails ({len(hits)}) for topic modeling. Need at least 5."}

        kw_stop = self._keyword_stopwords(keyword)
        # Keep each email paired with its tokens so dominant topics map back
        # to the right email ID after short documents are filtered out.
        pairs = [(e, self._tokenize(e.get("body", ""), kw_stop)) for e in hits]
        pairs = [(e, toks) for e, toks in pairs if len(toks) >= 5]
        if len(pairs) < 5:
            return {"error": f"Too few substantive emails ({len(pairs)}) for topic modeling. Need at least 5."}

        dictionary = corpora.Dictionary(toks for _, toks in pairs)
        dictionary.filter_extremes(no_below=2, no_above=0.6)
        if len(dictionary) < num_topics:
            return {"error": "Vocabulary too small after filtering for topic modeling. Try a broader keyword."}
        bow_corpus = [dictionary.doc2bow(toks) for _, toks in pairs]

        lda = LdaModel(
            corpus=bow_corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
        )

        topics = []
        for idx in range(num_topics):
            terms = lda.show_topic(idx, topn=8)
            words = [w for w, _ in terms]
            topics.append({
                "id":             idx,
                "label":          f"Topic {idx + 1}",
                "words":          words,
                "weights":        [round(float(p), 4) for _, p in terms],
                "interpretation": (AITopicInterpreter.interpret_topic(words)
                                   if interpret_topics else None),
            })

        doc_topics = []
        for (email, _), bow in zip(pairs, bow_corpus):
            dist = lda.get_document_topics(bow)
            if dist:
                topic_id, prob = max(dist, key=lambda x: x[1])
                doc_topics.append({
                    "email_id":       email["id"],
                    "dominant_topic": int(topic_id),
                    "confidence":     round(float(prob) * 100, 1),
                })

        return {
            "keyword":         keyword,
            "emails_analysed": len(pairs),
            "num_topics":      num_topics,
            "topics":          topics,
            "doc_topics":      doc_topics,
        }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — DIACHRONIC TIMELINE  (Pandas time-series)
# ─────────────────────────────────────────────────────────────────────────────
class DiachronicTimeline(CorpusProcessor):
    """
    Track syntactic hedging / logistic-verb / flagged-term density over time.

    order="date"   → emails with a recoverable date (see estimate_email_date),
                     sorted chronologically. Falls back to corpus order when
                     fewer than `min_dated` emails carry a date.
    order="corpus" → every matching email in the clean corpus's order (the
                     annotated file itself is sorted by flag count, which made
                     the old index-based chart a ranking, not a timeline).
    """

    def analyse(self, keyword: str, annotated_corpus=None, whole_word: bool = True,
                window: int = 10, order: str = "date", min_dated: int = 5) -> dict:
        if not _PD_AVAILABLE:
            raise RuntimeError("Pandas is not installed. Run: pip install pandas")

        if annotated_corpus is None:
            annotated_corpus = self.load_annotated()
        hits = self.query_by_keyword(annotated_corpus, keyword, whole_word)

        if not hits:
            return {"error": f"No emails found for keyword '{keyword}'."}

        rank = {e.get("id"): i for i, e in enumerate(self.load_clean())}
        rows = []
        for e in hits:
            body  = e.get("body", "")
            total = max(len(tokenize_words(body)), 1)
            rows.append({
                "email_id":         e["id"],
                "date":             estimate_email_date(body),
                "corpus_rank":      rank.get(e["id"], len(rank)),
                "sender":           e.get("sender", ""),
                "subject":          e.get("subject", ""),
                "hedge_density":    round(count_hedges(body) / total * 100, 3),
                "logistic_density": round(count_logistic_verbs(body) / total * 100, 3),
                "flag_count":       e.get("flag_count", 0),
                "body_length":      total,
            })

        df = pd.DataFrame(rows)
        dated_count = int(df["date"].notna().sum())
        if order == "date" and dated_count >= min_dated:
            df = df[df["date"].notna()].sort_values(["date", "corpus_rank"])
        else:
            order = "corpus"
            df = df.sort_values("corpus_rank")
        df = df.reset_index(drop=True)
        df.insert(0, "index", range(len(df)))
        df["date"] = pd.to_datetime(df["date"])

        df["hedge_rolling"]    = df["hedge_density"].rolling(window, min_periods=1).mean().round(3)
        df["logistic_rolling"] = df["logistic_density"].rolling(window, min_periods=1).mean().round(3)

        return {
            "keyword":              keyword,
            "order":                order,
            "total_emails":         len(df),
            "matched_emails":       len(rows),
            "dated_emails":         dated_count,
            "date_range":           ((df["date"].min().date().isoformat(), df["date"].max().date().isoformat())
                                     if order == "date" else None),
            "window":               window,
            "timeline":             df.to_dict(orient="records"),
            "peak_hedge_email":     df.loc[df["hedge_density"].idxmax()].to_dict(),
            "peak_logistic_email":  df.loc[df["logistic_density"].idxmax()].to_dict(),
            "avg_hedge_density":    round(float(df["hedge_density"].mean()), 3),
            "avg_logistic_density": round(float(df["logistic_density"].mean()), 3),
        }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — THREAT ASSESSOR  (HuggingFace zero-shot)
# ─────────────────────────────────────────────────────────────────────────────
FORENSIC_LABELS = [
    "Logistical Coordination",
    "Veiled Coercion",
    "Social Grooming",
    "Financial Transaction",
    "Routine Communication",
    "Information Suppression",
    "Coded Language Usage",
]

# Labels that describe benign intent — a high score on these is not a threat.
BENIGN_LABELS = {"Routine Communication"}


class ThreatAssessor(CorpusProcessor):
    """Zero-shot intent classification using HuggingFace Transformers."""

    @classmethod
    def get_pipeline(cls):
        return get_zero_shot_pipeline()

    @staticmethod
    def _validate_labels(labels) -> List[str]:
        labels = [l.strip() for l in (labels or FORENSIC_LABELS) if l and l.strip()]
        labels = list(dict.fromkeys(labels))  # de-duplicate, keep order
        if len(labels) < 2:
            raise ValueError("Provide at least two distinct labels to compare.")
        return labels

    @staticmethod
    def _format(result: dict) -> dict:
        scores = sorted(
            ({"label": l, "score": round(s * 100, 1)}
             for l, s in zip(result["labels"], result["scores"])),
            key=lambda x: x["score"], reverse=True,
        )
        return {
            "top_label":  scores[0]["label"],
            "top_score":  scores[0]["score"],
            "all_scores": scores,
        }

    def assess_many(self, bodies: Sequence[str], labels=None, batch_size: int = 8,
                    progress: Optional[Callable[[int, int], None]] = None) -> List[dict]:
        """Classify many emails with batched inference; `progress(done, total)` is optional."""
        labels = self._validate_labels(labels)
        snippets = [self.clean_text(b)[:512] for b in bodies]
        results: List[dict] = []
        for start in range(0, len(snippets), batch_size):
            chunk = snippets[start:start + batch_size]
            results.extend(self._format(r) for r in _zero_shot(chunk, labels, batch_size))
            if progress:
                progress(len(results), len(snippets))
        return results

    def assess_email(self, email_body: str, labels=None) -> dict:
        return self.assess_many([email_body], labels)[0]

    def search_and_assess(self, keyword: str, top_n: int = 10, corpus=None,
                          whole_word: bool = True, labels=None) -> list:
        if corpus is None:
            corpus = self.load_clean()
        hits = self.query_by_keyword(corpus, keyword, whole_word)[:top_n]
        assessments = self.assess_many([e.get("body", "") for e in hits], labels)
        return [
            {
                "email_id":     e["id"],
                "sender":       e.get("sender", ""),
                "subject":      e.get("subject", ""),
                "body_snippet": e.get("body", "")[:300],
                "body":         e.get("body", ""),
                **a,
            }
            for e, a in zip(hits, assessments)
        ]
