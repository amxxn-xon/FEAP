"""
engine.py — Forensic Entity-Action Profiler
Core OOP engine: CorpusProcessor base class + 5 analytical subclasses.

Lemalock Hypothesis — Computational Pragmatics Framework
Author: Ameen K.P | EFLU

IMPROVEMENTS:
- AI-powered dynamic wordlist expansion (replaces static dictionaries)
- AI-powered LDA topic interpretation
- Larger email batch support
- Node click data returned for spiderweb overlay
- Email lookup by ID for overlays
"""

import json
import re
import os
from collections import defaultdict, Counter
from typing import List, Dict, Any, Counter as TypingCounter

# ── Optional heavy imports (graceful fallback) ───────────────────────────────
try:
    import spacy  # type: ignore
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    import nltk  # type: ignore
    from nltk import ne_chunk, pos_tag as nltk_pos_tag, word_tokenize
    from nltk.tree import Tree  # type: ignore
    _NLTK_AVAILABLE = True

    # ── Cloud Readiness: Automatic NLTK Download ──────────────────────────────
    def _setup_nltk():
        for res in ['punkt', 'words', 'maxent_ne_chunker', 'averaged_perceptron_tagger']:
            try:
                nltk.data.find(res)
            except LookupError:
                nltk.download(res, quiet=True)
            except Exception:
                pass # Fallback for unexpected path issues

    _setup_nltk()

except ImportError:
    _NLTK_AVAILABLE = False

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
    from gensim.models import LdaModel
    from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOP  # type: ignore
    _GENSIM_AVAILABLE = True
except ImportError:
    _GENSIM_AVAILABLE = False

try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(__file__)
_CLEAN_PATH      = os.path.join(_BASE, "data", "epstein_clean_data.json")
_ANNOTATED_PATH  = os.path.join(_BASE, "data", "epstein_annotated.json")

# ─────────────────────────────────────────────────────────────────────────────
# STATIC WORDLISTS (Fallback — used when AI expansion is unavailable)
# These are kept as seeds; the AIWordlistExpander enriches them at runtime.
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
    'kind of', 'I think', 'I believe', 'I suppose', 'presumably'
}

LOGISTIC_VERBS = {
    'schedule', 'transport', 'arrange', 'coordinate', 'deliver', 'send',
    'move', 'transfer', 'book', 'reserve', 'confirm', 'organize',
    'prepare', 'handle', 'manage', 'pick up', 'drop off', 'bring', 'provide'
}

# ─────────────────────────────────────────────────────────────────────────────
# AI WORDLIST EXPANDER
# Uses the HuggingFace zero-shot pipeline to dynamically score & expand
# word membership in forensic categories — replaces hard-coded dictionaries.
# ─────────────────────────────────────────────────────────────────────────────
class AIWordlistExpander:
    """
    Uses a zero-shot NLI model to dynamically decide whether corpus words
    belong to forensic categories (food cipher, euphemism, logistical verb,
    hedge). Much more accurate than static dictionaries.
    """
    _pipeline = None
    # Cache: word → {category: bool}
    _cache = {}

    CATEGORIES = {
        "food_code": "This word is used as a food-related code word or culinary euphemism in illicit communications",
        "euphemism": "This word is used as a euphemism or coded language to conceal illicit activity",
        "logistic":  "This word describes a logistical coordination action like scheduling, transporting, or arranging",
        "hedge":     "This word is a modal hedge that reduces commitment or obscures intent in written communication",
    }

    @classmethod
    def _get_pipeline(cls):
        if cls._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore
                cls._pipeline = pipeline(
                    "zero-shot-classification",
                    model="cross-encoder/nli-MiniLM2-L6-H768",
                    device=-1,
                )
            except Exception:
                return None
        return cls._pipeline

    @classmethod
    def is_member(cls, word: str, category: str, threshold: float = 0.65) -> bool:
        """
        Returns True if the AI model scores `word` as belonging to `category`
        above the threshold. Falls back to static wordlists if model unavailable.
        """
        cache_key = f"{word}::{category}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        pipe = cls._get_pipeline()
        if pipe is None:
            # Graceful fallback to static lists
            result = cls._static_fallback(word, category)
            cls._cache[cache_key] = result
            return result

        label_text = cls.CATEGORIES.get(category, category)
        try:
            out = pipe(
                f'The word "{word}" appears in an email.',
                candidate_labels=[label_text, "This word has no special significance"],
                multi_label=False,
            )
            score = out["scores"][0] if out["labels"][0] == label_text else out["scores"][1]
            result = score >= threshold
        except Exception:
            result = cls._static_fallback(word, category)

        cls._cache[cache_key] = result
        return result

    @classmethod
    def _static_fallback(cls, word: str, category: str) -> bool:
        w = word.lower()
        if category == "food_code":   return w in FOOD_CODE_WORDS
        if category == "euphemism":   return w in EUPHEMISM_WORDS
        if category == "logistic":    return w in LOGISTIC_VERBS
        if category == "hedge":       return w in HEDGE_WORDS
        return False

    @classmethod
    def expand_corpus_wordlist(cls, corpus_words: List[str], category: str,
                               top_n: int = 40) -> set:
        """
        Scan the most frequent corpus words and return those the AI classifies
        as belonging to `category`. Merges with the static seed set.
        """
        static_bases = {
            "food_code": set(FOOD_CODE_WORDS),
            "euphemism": set(EUPHEMISM_WORDS),
            "logistic":  set(LOGISTIC_VERBS),
            "hedge":     set(HEDGE_WORDS),
        }
        base = static_bases.get(category, set())
        dynamic = set()
        for word in corpus_words[:top_n]:  # type: ignore
            if word not in base and cls.is_member(word, category):
                dynamic.add(word)
        return base | dynamic

    @classmethod
    def get_dynamic_wordlists(cls, corpus) -> dict:
        """
        Build all four dynamic wordlists from corpus vocabulary.
        Returns dict with keys: food_code, euphemism, logistic, hedge.
        """
        # Collect top 80 frequent corpus words (excluding very common stop words)
        stop = {'the','a','an','is','it','in','on','at','to','for','of','and',
                'or','but','not','this','that','with','from','by','be','as',
                'are','was','were','have','has','had','will','i','you','he',
                'she','we','they','me','him','her','us','them','my','your'}
        all_words = []
        for e in corpus[:500]:
            tokens = re.findall(r'\b[a-z]{3,}\b', e.get("body","").lower())
            all_words.extend(t for t in tokens if t not in stop)
        top_words = [w for w, _ in Counter(all_words).most_common(100)]

        return {
            "food_code": cls.expand_corpus_wordlist(top_words, "food_code"),
            "euphemism": cls.expand_corpus_wordlist(top_words, "euphemism"),
            "logistic":  cls.expand_corpus_wordlist(top_words, "logistic"),
            "hedge":     cls.expand_corpus_wordlist(top_words, "hedge"),
        }

# ─────────────────────────────────────────────────────────────────────────────
# AI TOPIC INTERPRETER
# Calls HuggingFace to generate a human-readable description for each LDA topic
# ─────────────────────────────────────────────────────────────────────────────
class AITopicInterpreter:
    """
    Uses zero-shot classification to match LDA topic word clusters to
    a set of forensic semantic categories, producing interpretable labels
    and descriptions.
    """
    _pipeline = None

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

    @classmethod
    def _get_pipeline(cls):
        if cls._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore
                cls._pipeline = pipeline(
                    "zero-shot-classification",
                    model="cross-encoder/nli-MiniLM2-L6-H768",
                    device=-1,
                )
            except Exception:
                return None
        return cls._pipeline

    @classmethod
    def interpret_topic(cls, words, weights) -> dict:
        """
        Given LDA topic words + weights, return:
          - best_label: matched forensic category
          - confidence: score 0–100
          - description: plain-English explanation
          - forensic_significance: brief forensic note
        """
        pipe = cls._get_pipeline()
        if pipe is None or not words:
            return cls._fallback_interpretation(words)

        # Build a natural-language description of the topic for classification
        top_words = ", ".join(words[:6])
        text = f"An email cluster whose most distinctive words are: {top_words}."

        try:
            result = pipe(text, candidate_labels=cls.FORENSIC_TOPIC_LABELS, multi_label=False)  # type: ignore
            best_label = result["labels"][0]
            confidence = round(result["scores"][0] * 100, 1)

            description = cls._build_description(best_label, words, confidence)
            significance = cls._forensic_note(best_label)

            return {
                "best_label": best_label,
                "confidence": confidence,
                "description": description,
                "forensic_significance": significance,
                "ai_powered": True,
            }
        except Exception:
            return cls._fallback_interpretation(words)

    @classmethod
    def _build_description(cls, label: str, words, confidence: float) -> str:
        top = ", ".join(f'"{w}"' for w in words[:4])
        conf_text = "strongly" if confidence > 70 else "moderately" if confidence > 45 else "tentatively"
        return (
            f"This topic cluster {conf_text} matches the pattern of "
            f"<strong>{label}</strong>. "
            f"Key signal words include {top}. "
            f"AI confidence: {confidence}%."
        )

    @classmethod
    def _forensic_note(cls, label: str) -> str:
        notes = {
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
        return notes.get(label, "ℹ️ Review manually for contextual significance.")

    @classmethod
    def _fallback_interpretation(cls, words) -> dict:
        """Rule-based fallback when transformer is unavailable."""
        food_signals    = {'food','pizza','chicken','dinner','lunch','menu','cake'}
        logistic_signals = {'schedule','transport','arrange','book','travel','flight'}
        coerce_signals  = {'must','need','expect','required','demand','force'}

        top = set(words[:6])
        if top & food_signals:
            label = "Coded food/service euphemisms for illicit activity"
        elif top & logistic_signals:
            label = "Logistical coordination and travel planning"
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
    _annotated_cache = None

    # ── Loading ──────────────────────────────────────────────────────────────
    @classmethod
    def load_clean(cls):
        if cls._clean_cache is None:
            with open(_CLEAN_PATH, "r", encoding="utf-8") as f:
                cls._clean_cache = json.load(f)
        return cls._clean_cache

    @classmethod
    def load_annotated(cls):
        if cls._annotated_cache is None:
            with open(_ANNOTATED_PATH, "r", encoding="utf-8") as f:
                cls._annotated_cache = json.load(f)
        return cls._annotated_cache

    @classmethod
    def get_email_by_id(cls, email_id: str):
        """Look up a full email record by its ID from either corpus."""
        corpus = cls.load_clean()
        for e in corpus:
            if e.get("id") == email_id:
                return e
        # Also check annotated
        try:
            ann = cls.load_annotated()
            for e in ann:
                if e.get("id") == email_id:
                    return e
        except Exception:
            pass
        return None

    # ── Functional querying ──────────────────────────────────────────────────
    @staticmethod
    def query_by_keyword(corpus, keyword: str) -> List[Dict[str, Any]]:
        kw = keyword.lower()
        return list(filter(
            lambda e: kw in e.get("body", "").lower()
                      or kw in e.get("subject", "").lower(),
            corpus
        ))

    @staticmethod
    def extract_sentences_with_keyword(emails, keyword: str) -> List[Dict[str, Any]]:
        kw = keyword.lower()
        result = []
        for e in emails:
            body = e.get("body", "")
            sentences = re.split(r'(?<=[.!?])\s+', body)
            for s in sentences:
                if kw in s.lower():
                    result.append({
                        "id": e["id"], "sender": e["sender"],
                        "subject": e["subject"], "sentence": s.strip()
                    })
        return result

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[\u2588]+', '[REDACTED]', text)
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
    def __init__(self):
        self._nlp = None

    def _get_nlp(self):
        if self._nlp is None:
            if not _SPACY_AVAILABLE:
                raise RuntimeError(
                    "spaCy is not installed. Run: pip install spacy && "
                    "python -m spacy download en_core_web_sm"
                )
            import spacy  # type: ignore
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def analyse(self, keyword: str, max_sentences: int = 30, corpus=None) -> dict:
        if corpus is None: corpus = self.load_clean()
        hits       = self.query_by_keyword(corpus, keyword)
        sentences  = self.extract_sentences_with_keyword(hits, keyword)[:max_sentences]  # type: ignore

        results = []
        root_verbs: TypingCounter[str] = Counter()
        modifiers: TypingCounter[str] = Counter()
        hedge_count: int = 0
        logistic_count: int = 0

        nlp = self._get_nlp()
        texts = [item["sentence"] for item in sentences]
        docs = list(nlp.pipe(texts, batch_size=50))

        for item, doc in zip(sentences, docs):

            kw_tokens = [t for t in doc if keyword.lower() in t.text.lower()]
            if not kw_tokens:
                continue

            root = next((t for t in doc if t.dep_ == "ROOT"), None)
            if root:
                root_verbs.update([root.lemma_])
                if root.lemma_.lower() in LOGISTIC_VERBS:
                    logistic_count += 1  # type: ignore

            for kw_tok in kw_tokens:
                for child in kw_tok.children:
                    modifiers.update([child.text.lower()])

            lower_sent = item["sentence"].lower()  # type: ignore
            if any(h in lower_sent for h in HEDGE_WORDS):
                hedge_count += 1  # type: ignore

            edges = [
                {"from": t.head.text, "to": t.text, "dep": t.dep_}
                for t in doc if t.dep_ != "punct"
            ]
            tokens_info = [
                {
                    "text": t.text, "pos": t.pos_, "dep": t.dep_,
                    "head": t.head.text,
                    "is_keyword": keyword.lower() in t.text.lower()
                }
                for t in doc
            ]
            import pandas as pd  # type: ignore
            edges_df = pd.DataFrame([
                {"from": e["from"], "to": e["to"], "dep": e["dep"]} for e in edges  # type: ignore
            ])
            
            item_data = {
                "sentence":  item["sentence"],  # type: ignore
                "email_id":  item["id"],        # type: ignore
                "sender":    item["sender"],    # type: ignore
                "subject":   item["subject"],   # type: ignore
                "edges":     edges,
                "tokens":    tokens_info,
                "root":      getattr(root, "text", "") if root else "",
            }
            results.append(item_data)  # type: ignore

        total_hits   = len(hits)
        hedge_pct    = round(float(hedge_count    / max(len(sentences), 1) * 100), 1)  # type: ignore
        logistic_pct = round(float(logistic_count / max(len(sentences), 1) * 100), 1)  # type: ignore

        return {
            "keyword":                  keyword,
            "total_emails_with_keyword": total_hits,
            "sentences_analysed":       len(sentences),
            "hedge_percentage":         hedge_pct,
            "logistic_verb_percentage": logistic_pct,
            "top_root_verbs":           root_verbs.most_common(10),  # type: ignore
            "top_modifiers":            modifiers.most_common(10),  # type: ignore
            "sentence_details":         results,
        }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — ENTITY SPIDERWEB  (spaCy NER + NetworkX)
# ─────────────────────────────────────────────────────────────────────────────
class EntitySpiderweb(CorpusProcessor):
    """Named-entity co-occurrence graph centred on a keyword."""
    def __init__(self):
        self._nlp = None

    def _get_nlp(self):
        if self._nlp is None:
            if not _SPACY_AVAILABLE:
                raise RuntimeError(
                    "spaCy is not installed. Run: pip install spacy && "
                    "python -m spacy download en_core_web_sm"
                )
            import spacy  # type: ignore
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def analyse(self, keyword: str, max_emails: int = 150, corpus=None) -> dict:
        """
        max_emails increased to 150 by default; caller can push higher via slider.
        Now also returns email_entity_map so the UI can open email overlays
        when a node is clicked.
        """
        if corpus is None: corpus = self.load_clean()
        hits   = self.query_by_keyword(corpus, keyword)[:max_emails]  # type: ignore

        edge_weights: TypingCounter[tuple] = Counter()
        node_types = {}
        # node_id → list of email dicts that contain this entity
        node_email_map = defaultdict(list)
        email_entity_map = []

        nlp = self._get_nlp()
        texts = [email.get("body", "")[:2000] for email in hits]
        docs = list(nlp.pipe(texts, batch_size=50))

        for email, doc in zip(hits, docs):
            entities = []
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "FAC", "LOC"):
                    label = ent.label_
                    if label == "ORG": label = "ORGANIZATION"
                    if label == "LOC": label = "LOCATION"
                    if label == "FAC": label = "FACILITY"
                    entities.append({"name": ent.text.strip(), "type": label})

            names = list({e["name"] for e in entities if len(e["name"]) > 2})

            for e in entities:
                node_types[e["name"]] = e["type"]
                node_email_map[e["name"]].append({  # type: ignore
                    "id":      email["id"],
                    "sender":  email.get("sender", ""),
                    "subject": email.get("subject", ""),
                    "body":    email.get("body", ""),
                })

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    pair = tuple(sorted([names[i], names[j]]))
                    edge_weights[pair] += 1  # type: ignore

            if names:
                email_entity_map.append({"id": email["id"], "entities": names})

        nodes = [{"id": name, "type": ntype} for name, ntype in node_types.items()]
        edges = [
            {"source": a, "target": b, "weight": w}
            for (a, b), w in edge_weights.most_common(80)  # type: ignore
        ]

        # Deduplicate node_email_map entries by email id
        for node_id in list(node_email_map.keys()):  # type: ignore
            seen = set()
            unique = []
            for em in node_email_map[node_id]:  # type: ignore
                if em["id"] not in seen:
                    seen.add(em["id"])
                    unique.append(em)
            node_email_map[node_id] = unique  # type: ignore

        return {
            "keyword":          keyword,
            "total_emails":     len(hits),
            "nodes":            nodes,
            "edges":            edges,
            "node_email_map":   dict(node_email_map),   # NEW: for click overlays
            "email_entity_map": email_entity_map,
            "top_entities":     Counter({
                n["id"]: sum(
                    e["weight"]
                    for e in edges
                    if e["source"] == n["id"] or e["target"] == n["id"]
                )
                for n in nodes
            }).most_common(15),
        }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — HIDDEN CONTEXT DECODER  (Gensim LDA + AI topic interpretation)
# ─────────────────────────────────────────────────────────────────────────────
class ContextDecoder(CorpusProcessor):
    """
    Unsupervised LDA topic modeling on emails filtered by keyword.
    Now enriched with AI-powered topic descriptions via AITopicInterpreter.
    """
    _STOP = {
        'the', 'a', 'an', 'is', 'it', 'in', 'on', 'at', 'to', 'for', 'of',
        'and', 'or', 'but', 'not', 'this', 'that', 'with', 'from', 'by',
        'be', 'as', 'are', 'was', 'were', 'have', 'has', 'had', 'will',
        'would', 'could', 'should', 'may', 'might', 'i', 'you', 'he', 'she',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
        'his', 'its', 'our', 'their', 'do', 'did', 'does', 'been', 'being',
        'can', 'all', 'if', 'about', 'up', 'out', 'so', 'also', 'just',
        'no', 'than', 'then', 'when', 'what', 'there', 'which', 'who',
        'how', 'any', 'some', 'more', 'very', 'get', 'got', 'go', 'will',
        'let', 'know', 'think', 'want', 'like', 'well', 'back', 'one',
        'time', 'see', 'please', 'thank', 're', 'sent', 'subject', 'dear',
        'regards', 'best', 'kind', 'hello', 'hi', 'email'
    }

    def _tokenize(self, text: str) -> list:
        tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [t for t in tokens if t not in self._STOP]

    def analyse(self, keyword: str, num_topics: int = 5,
                interpret_topics: bool = True, corpus=None) -> dict:
        if not _GENSIM_AVAILABLE:
            raise RuntimeError("Gensim is not installed. Run: pip install gensim")

        if corpus is None: corpus = self.load_clean()
        hits   = self.query_by_keyword(corpus, keyword)

        if len(hits) < 5:
            return {"error": f"Too few emails ({len(hits)}) for topic modeling. Need at least 5."}

        docs       = [self._tokenize(e.get("body", "")) for e in hits]
        docs       = [d for d in docs if len(d) >= 5]

        dictionary = corpora.Dictionary(docs)
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        bow_corpus = [dictionary.doc2bow(d) for d in docs]

        lda = LdaModel(
            corpus=bow_corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
        )

        topics = []
        for idx, topic in lda.print_topics(num_topics, num_words=8):
            words   = re.findall(r'"([^"]+)"', topic)
            weights = list(map(float, re.findall(r'(\d+\.\d+)\*', topic)))

            topic_dict = {
                "id":     idx,
                "label":  f"Topic {idx + 1}",
                "words":  words,
                "weights": weights,
                "interpretation": None,
            }

            # AI interpretation
            if interpret_topics:
                try:
                    topic_dict["interpretation"] = AITopicInterpreter.interpret_topic(
                        words, weights
                    )
                    interp = topic_dict["interpretation"]
                    if isinstance(interp, dict) and interp.get("best_label"):  # type: ignore
                        topic_dict["label"] = f"Topic {idx + 1}"
                except Exception:
                    pass

            topics.append(topic_dict)

        # Dominant topic per doc
        doc_topics = []
        for i, bow in enumerate(bow_corpus):
            dist = lda.get_document_topics(bow)
            if dist:
                dominant = max(dist, key=lambda x: x[1])
                doc_topics.append({
                    "email_id":       hits[i]["id"],
                    "dominant_topic": dominant[0],
                    "confidence":     round(dominant[1] * 100, 1),
                })

        return {
            "keyword":         keyword,
            "emails_analysed": len(docs),
            "num_topics":      num_topics,
            "topics":          topics,
            "doc_topics":      doc_topics,
        }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — DIACHRONIC TIMELINE  (Pandas time-series)
# ─────────────────────────────────────────────────────────────────────────────
class DiachronicTimeline(CorpusProcessor):
    """Track syntactic hedging / flagged-term density over time (by email index)."""
    def analyse(self, keyword: str, annotated_corpus=None) -> dict:
        if not _PD_AVAILABLE:
            raise RuntimeError("Pandas is not installed. Run: pip install pandas")

        if annotated_corpus is None:
            annotated_corpus = self.load_annotated()
        
        kw        = keyword.lower()
        hits      = [e for e in annotated_corpus if kw in e.get("body", "").lower()]

        if not hits:
            return {"error": f"No emails found for keyword '{keyword}'."}

        rows = []
        for i, e in enumerate(hits):
            body    = e.get("body", "")
            tokens  = body.lower().split()
            total   = max(len(tokens), 1)

            hedge_count    = sum(1 for t in tokens if t in HEDGE_WORDS)
            logistic_count = sum(1 for t in tokens if t in LOGISTIC_VERBS)
            flag_count     = e.get("flag_count", 0)

            rows.append({
                "index":            i,
                "email_id":         e["id"],
                "sender":           e.get("sender", ""),
                "subject":          e.get("subject", ""),
                "body":             e.get("body", ""),      # full body for overlay
                "hedge_density":    round(hedge_count    / total * 100, 3),  # type: ignore
                "logistic_density": round(logistic_count / total * 100, 3),  # type: ignore
                "flag_count":       flag_count,
                "body_length":      len(tokens),
            })

        df = pd.DataFrame(rows)
        df["hedge_rolling"]    = df["hedge_density"].rolling(10, min_periods=1).mean().round(3)  # type: ignore
        df["logistic_rolling"] = df["logistic_density"].rolling(10, min_periods=1).mean().round(3)  # type: ignore

        # Exclude 'body' from the serialised output to keep payload small;
        # it's accessible via get_email_by_id() for overlays.
        timeline_records = df.drop(columns=["body"]).to_dict(orient="records")

        return {
            "keyword":               keyword,
            "total_emails":          len(rows),
            "timeline":              timeline_records,
            "peak_hedge_email":      df.loc[df["hedge_density"].idxmax()].to_dict(),
            "peak_logistic_email":   df.loc[df["logistic_density"].idxmax()].to_dict(),
            "avg_hedge_density":     round(df["hedge_density"].mean(), 3),
            "avg_logistic_density":  round(df["logistic_density"].mean(), 3),
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

class ThreatAssessor(CorpusProcessor):
    """Zero-shot intent classification using HuggingFace Transformers."""
    _pipeline = None

    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore
                cls._pipeline = pipeline(
                    "zero-shot-classification",
                    model="cross-encoder/nli-MiniLM2-L6-H768",
                    device=-1,
                )
            except ImportError:
                raise RuntimeError(
                    "transformers or torch not installed. "
                    "Run: pip install transformers torch"
                )
            except Exception as e:
                raise RuntimeError(f"Error initializing NLP pipeline: {str(e)}")
        return cls._pipeline

    def assess_email(self, email_body: str, labels = None) -> dict:
        if labels is None:
            labels = FORENSIC_LABELS

        pipe    = self.get_pipeline()
        snippet = email_body[:512]  # type: ignore
        result  = pipe(snippet, candidate_labels=labels, multi_label=False)

        scores = [
            {"label": l, "score": round(s * 100, 1)}
            for l, s in zip(result["labels"], result["scores"])
        ]
        scores.sort(key=lambda x: x["score"], reverse=True)

        return {
            "top_label":  scores[0]["label"],
            "top_score":  scores[0]["score"],
            "all_scores": scores,
        }

    def search_and_assess(self, keyword: str, top_n: int = 10, corpus=None) -> list:
        if corpus is None: corpus = self.load_clean()
        hits       = self.query_by_keyword(corpus, keyword)[:top_n]  # type: ignore
        results = []
        for e in hits:
            assessment = self.assess_email(e["body"])
            results.append({
                "email_id":     e["id"],
                "sender":       e["sender"],
                "subject":      e["subject"],
                "body_snippet": e["body"][:300],
                "body":         e["body"],
                **assessment,
            })
        return results
