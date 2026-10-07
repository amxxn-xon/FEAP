"""
app.py — Forensic Entity-Action Profiler
Streamlit Dashboard | Computational Pragmatics Framework
Author: Ameen K.P | EFLU

Notes:
- Analysis results are cached per keyword/setting, so interacting with a
  widget no longer re-runs spaCy, NER or LDA from scratch.
- All corpus text is HTML-escaped before rendering (some emails contain
  raw "<" fragments that previously swallowed parts of the page).
- Every result table can be downloaded as CSV.
"""

import html
import os
import re
import sys
from collections import Counter

import streamlit as st  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (  # type: ignore
    CorpusProcessor, SyntacticXRay, EntitySpiderweb,
    ContextDecoder, DiachronicTimeline, ThreatAssessor,
    AIWordlistExpander,
    FORENSIC_LABELS, BENIGN_LABELS, FOOD_CODE_WORDS, EUPHEMISM_WORDS,
    LOGISTIC_VERBS, keyword_pattern, has_hedge, find_hedges,
    count_logistic_verbs, tokenize_words,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forensic Entity-Action Profiler",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# HTML RENDERING
# ─────────────────────────────────────────────────────────────────────────────
def render_html(markup: str, target=None) -> None:
    """Render raw HTML. Lines are stripped and blank lines dropped so Markdown
    never ends the HTML block early or turns indented lines into code blocks."""
    tidy = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    (target or st).markdown(tidy, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #f8fafc;
    --bg-card: #ffffff;
    --bg-glass: rgba(255,255,255,0.7);
    --border: rgba(99,179,237,0.25);
    --accent: #3182ce;
    --accent2: #e53e3e;
    --accent3: #38a169;
    --accent4: #805ad5;
    --accent5: #d69e2e;
    --text-primary: #1e293b;
    --text-muted: #64748b;
    --text-mono: #0d9488;
}

header,[data-testid="stHeader"],.stAppHeader{background-color:transparent!important;background:transparent!important;}
footer{visibility:hidden;}
[data-testid="stAppViewContainer"]{padding-top:0px!important;}
[data-testid="stSidebarContent"]{padding-top:0px!important;margin-top:-2.5rem!important;}
[data-testid="stSidebarNav"]{padding-top:0px!important;}
[data-testid="stMainBlockContainer"],.main .block-container{padding-top:0.5rem!important;padding-bottom:0rem!important;margin-top:-0.5rem!important;}
section[data-testid="stSidebar"]+div{padding-top:0px!important;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg-primary)!important;color:var(--text-primary)!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#f1f5f9 0%,#ffffff 100%)!important;border-right:1px solid var(--border);}

.metric-card{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:18px 22px;
    box-shadow:0 4px 6px -1px rgba(0,0,0,0.05),0 2px 4px -1px rgba(0,0,0,0.03);margin-bottom:12px;transition:all 0.2s;}
.metric-card:hover{border-color:var(--accent);transform:translateY(-1px);box-shadow:0 10px 15px -3px rgba(0,0,0,0.1);}
.metric-value{font-size:2.2rem;font-weight:700;color:var(--accent);line-height:1;}
.metric-label{font-size:0.78rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.08em;margin-top:4px;}

.section-header{display:flex;align-items:center;gap:10px;padding:14px 0 8px 0;border-bottom:2px solid var(--border);margin-bottom:22px;}
.section-header h2{font-size:1.25rem;font-weight:700;color:var(--text-primary);margin:0;letter-spacing:-0.02em;}
.section-badge{margin-left:auto;font-size:0.7rem;font-weight:600;color:var(--accent);background:#ebf8ff;
    border:1px solid #bee3f8;border-radius:999px;padding:3px 10px;letter-spacing:0.03em;}

.email-card{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:16px 20px;
    margin-bottom:12px;font-size:0.88rem;color:var(--text-primary);box-shadow:0 1px 3px rgba(0,0,0,0.05);transition:all 0.2s;}
.email-card:hover{border-color:var(--accent);background:#f1f5f9;}
.email-card .meta{color:var(--text-muted);font-size:0.78rem;margin-bottom:6px;}
.email-card .sender{color:var(--accent);font-weight:600;}
.email-card .subject{color:var(--accent4);font-weight:500;}
.email-card .body-snippet{color:var(--text-primary);line-height:1.6;word-break:break-word;}
.email-id{color:var(--text-mono);font-family:'JetBrains Mono',monospace;font-size:0.72rem;}
.empty-state{text-align:center;padding:32px;color:#718096;}

mark.kw{background:rgba(251,211,141,0.5);color:#92400e;border-radius:3px;padding:0 3px;font-weight:600;}

.overlay-meta{font-size:0.8rem;color:var(--text-muted);line-height:2.2;margin-bottom:14px;
    padding:12px 16px;background:#f8fafc;border-radius:8px;border:1px solid var(--border);}
.overlay-body{background:#f8fafc;border:1px solid var(--border);border-radius:10px;padding:20px;
    font-size:0.88rem;line-height:1.85;color:var(--text-primary);max-height:420px;overflow-y:auto;
    white-space:pre-wrap;word-break:break-word;}
.full-email{margin-top:12px;border-top:1px solid var(--border);padding-top:10px;}
.full-email summary{cursor:pointer;color:var(--accent);font-weight:600;font-size:0.85rem;outline:none;}
.full-email .overlay-body{margin-top:8px;max-height:350px;}

/* Token pills */
.token-pill{display:inline-block;padding:3px 12px;border-radius:6px;font-size:0.75rem;
    font-weight:600;margin:3px 4px;font-family:'JetBrains Mono',monospace;}
.token-pill small{display:block;font-size:0.6rem;opacity:0.7;}
.pill-noun    {background:#ebf8ff;color:#3182ce;border:1px solid #bee3f8;}
.pill-verb    {background:#fff5f5;color:#e53e3e;border:1px solid #fed7d7;}
.pill-adj     {background:#f0fff4;color:#38a169;border:1px solid #c6f6d5;}
.pill-adv     {background:#faf5ff;color:#805ad5;border:1px solid #e9d8fd;}
.pill-keyword {background:#fffff0;color:#d69e2e;border:1px solid #fefcbf;font-weight:800;}
.pill-default {background:#f7fafc;color:#4a5568;border:1px solid #edf2f7;}

.score-bar-wrap{margin-bottom:12px;}
.score-label{font-size:0.85rem;font-weight:500;color:var(--text-primary);margin-bottom:4px;display:flex;justify-content:space-between;gap:8px;}
.score-bar{height:10px;border-radius:5px;background:#edf2f7;border:1px solid #e2e8f0;}
.score-fill{height:10px;border-radius:5px;}

.threat-badge{display:inline-block;padding:6px 16px;border-radius:8px;font-weight:800;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.02em;}
.threat-high  {background:#fff5f5;color:#e53e3e;border:1.5px solid #feb2b2;}
.threat-medium{background:#fffaf0;color:#dd6b20;border:1.5px solid #fbd38d;}
.threat-low   {background:#f0fff4;color:#38a169;border:1.5px solid #9ae6b4;}

.hero{background:linear-gradient(135deg,#f1f5f9 0%,#ffffff 100%);border:1px solid var(--border);
    border-radius:20px;padding:24px 45px 22px;margin-bottom:24px;box-shadow:0 4px 20px rgba(0,0,0,0.03);}
.hero-title{font-size:2.2rem;font-weight:800;color:var(--text-primary);letter-spacing:-0.04em;margin:0 0 2px 0;}
.hero-title span{color:var(--accent);}
.hero-sub{font-size:0.95rem;color:var(--text-muted);margin:0;line-height:1.7;font-weight:400;}

.module-intro{color:#718096;font-size:0.83rem;margin-bottom:18px;}
.module-intro .kw-ref{color:#d69e2e;font-weight:700;}
.panel-title{color:#a0aec0;font-weight:600;font-size:0.85rem;margin-bottom:8px;}

div[data-testid="stTabs"] button{color:var(--text-muted)!important;font-weight:500!important;}
div[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important;font-weight:700!important;}
.stTextInput>div>div>input{background:#ffffff!important;border:1px solid var(--border)!important;
    color:var(--text-primary)!important;border-radius:8px!important;font-size:0.95rem!important;}
.stSelectbox>div>div{background:#ffffff!important;border:1px solid var(--border)!important;border-radius:8px!important;}
.stButton>button{background:#3182ce!important;color:white!important;border:1px solid #2b6cb0!important;
    border-radius:8px!important;font-weight:700!important;padding:0.5rem 1.2rem!important;
    width:100%!important;box-shadow:0 1px 3px rgba(0,0,0,0.1)!important;transition:all 0.2s!important;margin-bottom:4px!important;}
.stButton>button *{color:white!important;}
.stButton>button:hover{background:#2b6cb0!important;box-shadow:0 4px 6px rgba(0,0,0,0.1)!important;transform:translateY(-1px);}
.stButton>button:active{transform:translateY(0);}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]>div{padding:0px!important;gap:2px!important;}
div[data-testid="stExpander"]{background:var(--bg-glass)!important;border:1px solid var(--border)!important;border-radius:10px!important;}
.stSpinner>div{border-top-color:var(--accent)!important;}

/* AI interpretation cards */
.ai-interp-card{background:linear-gradient(135deg,#ebf8ff,#f0fff4);border:1px solid #bee3f8;
    border-radius:10px;padding:14px 18px;margin-top:10px;font-size:0.82rem;line-height:1.7;text-align:left;}
.ai-badge{display:inline-block;background:#3182ce;color:white;font-size:0.65rem;font-weight:700;
    padding:2px 7px;border-radius:4px;letter-spacing:0.05em;margin-bottom:6px;vertical-align:middle;}
.ai-badge-fallback{background:#a0aec0;}

.progress-status{background:#f1f5f9;border:1px solid var(--border);border-radius:8px;
    padding:10px 16px;font-size:0.82rem;color:var(--text-muted);margin-bottom:12px;}

.node-hint{background:#fffff0;border:1px solid #fefcbf;border-radius:8px;
    padding:8px 14px;font-size:0.78rem;color:#92400e;margin-bottom:10px;}

.wl-tag{display:inline-block;padding:3px 10px;border-radius:6px;font-size:0.75rem;
    font-weight:600;margin:3px;background:#f7fafc;color:#4a5568;border:1px solid #edf2f7;}
.wl-tag.hedge{background:#fff5f5;color:#c53030;border-color:#fed7d7;}
.sidebar-label{color:#718096;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;margin:12px 0 8px 0;}
</style>
""")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def esc(value) -> str:
    """HTML-escape corpus/user text. `$` is escaped too so Streamlit's
    markdown never treats dollar amounts as LaTeX."""
    return html.escape(str(value), quote=True).replace("$", "&#36;")


def highlight_keyword(text: str, pattern: "re.Pattern[str]", max_len=None) -> str:
    """Escape `text` and wrap every keyword match in <mark>."""
    text = text or ""
    snippet = text if max_len is None else text[:max_len]
    out, last = [], 0
    for m in pattern.finditer(snippet):
        out.append(esc(snippet[last:m.start()]))
        out.append(f'<mark class="kw">{esc(m.group(0))}</mark>')
        last = m.end()
    out.append(esc(snippet[last:]))
    ellipsis = "…" if max_len is not None and len(text) > max_len else ""
    return "".join(out).replace("\n", "<br>") + ellipsis


_UPOS_CLASS = {"NOUN": "pill-noun", "PROPN": "pill-noun", "VERB": "pill-verb",
               "AUX": "pill-verb", "ADJ": "pill-adj", "ADV": "pill-adv"}


def pos_pill(text: str, pos: str, is_keyword: bool) -> str:
    if is_keyword:
        cls = "pill-keyword"
    else:
        # spaCy emits Universal POS tags; Penn prefixes are kept for safety.
        cls = _UPOS_CLASS.get(pos) or next(
            (c for p, c in (("NN", "pill-noun"), ("VB", "pill-verb"),
                            ("JJ", "pill-adj"), ("RB", "pill-adv")) if pos.startswith(p)),
            "pill-default")
    return f'<span class="token-pill {cls}">{esc(text)}<small>{esc(pos)}</small></span>'


def threat_badge(label: str, score: float) -> str:
    if label in BENIGN_LABELS:
        cls, icon = "threat-low", "✓"
    else:
        cls = "threat-high" if score > 60 else "threat-medium" if score > 35 else "threat-low"
        icon = "⚡"
    return f'<span class="threat-badge {cls}">{icon} {esc(label)} — {score}%</span>'


def score_bar(label: str, score: float, color: str = "#63b3ed", display=None) -> str:
    width = max(0.0, min(float(score), 100.0))
    shown = display if display is not None else f"{score}%"
    return f"""
    <div class="score-bar-wrap">
        <div class="score-label"><span>{esc(label)}</span><strong>{esc(shown)}</strong></div>
        <div class="score-bar">
            <div class="score-fill" style="width:{width}%;background:{color}"></div>
        </div>
    </div>"""


def section_header(icon: str, title: str, badge: str = "") -> None:
    badge_html = f'<span class="section-badge">{esc(badge)}</span>' if badge else ""
    render_html(f"""
    <div class="section-header">
        <span style="font-size:1.3rem">{icon}</span>
        <h2>{esc(title)}</h2>
        {badge_html}
    </div>""")


def metric_card(value, label: str, col=None) -> None:
    html_ = f"""
    <div class="metric-card">
        <div class="metric-value">{esc(value)}</div>
        <div class="metric-label">{esc(label)}</div>
    </div>"""
    render_html(html_, col)


def email_meta(email: dict, subject_len: int = 60) -> str:
    return (f'<code class="email-id">{esc(email.get("id", "N/A"))}</code> &nbsp;|&nbsp; '
            f'<span class="sender">{esc(email.get("sender") or "Unknown")}</span> &nbsp;|&nbsp; '
            f'<span class="subject">{esc((email.get("subject") or "(no subject)")[:subject_len])}</span>')


def email_card(email: dict, pattern, snippet_len: int = 300, extra_html: str = "",
               full_body_inline: bool = False) -> None:
    """Card with metadata + highlighted snippet; optionally an inline full-body toggle."""
    body = email.get("body", "")
    details = ""
    if full_body_inline and len(body) > snippet_len:
        details = (f'<details class="full-email"><summary>View full email</summary>'
                   f'<div class="overlay-body">{highlight_keyword(body, pattern)}</div></details>')
    render_html(f"""
    <div class="email-card">
        <div class="meta">{email_meta(email)}</div>
        {extra_html}
        <div class="body-snippet" style="margin-top:8px">{highlight_keyword(body, pattern, snippet_len)}</div>
        {details}
    </div>""")


def render_email_overlay(email: dict, pattern=None) -> None:
    """Expandable full-email view with metadata and highlighted body."""
    if not email:
        return
    eid  = email.get("id", "unknown")
    body = email.get("body") or "(no body)"
    with st.expander(f"📧 Open full email — {eid[:24]}", expanded=False):
        render_html(f"""
        <div class="overlay-meta">
            <strong>📌 ID:</strong> <code class="email-id">{esc(eid)}</code><br>
            <strong>👤 From:</strong> <span style="color:#3182ce">{esc(email.get("sender") or "Unknown")}</span><br>
            <strong>📝 Subject:</strong> <span style="color:#805ad5">{esc(email.get("subject") or "(no subject)")}</span>
        </div>
        <div class="overlay-body">{highlight_keyword(body, pattern) if pattern else esc(body)}</div>
        """)


def csv_download(rows, filename: str, label: str = "⬇️ Download CSV", key=None) -> None:
    import pandas as pd  # type: ignore
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        return
    st.download_button(label, df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv", key=key, on_click="ignore")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "keyword"


def words_regex(words) -> "re.Pattern[str]":
    alts = sorted((r"\s+".join(map(re.escape, w.split())) for w in words if w.strip()),
                  key=len, reverse=True)
    return re.compile(rf"(?<!\w)(?:{'|'.join(alts) or '(?!x)x'})(?!\w)", re.IGNORECASE)


TOPIC_COLOURS = ["#63b3ed", "#fc8181", "#68d391", "#b794f4", "#f6ad55",
                 "#4fd1c5", "#f687b3", "#ecc94b", "#9ae6b4", "#a3bffa"]

LABEL_COLOURS = {
    "Logistical Coordination":  "#63b3ed",
    "Veiled Coercion":          "#fc8181",
    "Social Grooming":          "#fbd38d",
    "Financial Transaction":    "#68d391",
    "Routine Communication":    "#a0aec0",
    "Information Suppression":  "#d6bcfa",
    "Coded Language Usage":     "#f6ad55",
}


def label_colour(label: str, idx: int = 0) -> str:
    return LABEL_COLOURS.get(label, TOPIC_COLOURS[idx % len(TOPIC_COLOURS)])


ENTITY_COLOURS = {
    "PERSON":       "#63b3ed",
    "ORGANIZATION": "#fc8181",
    "GPE":          "#68d391",
    "LOCATION":     "#68d391",
    "FACILITY":     "#fbd38d",
}

# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA & ANALYSES
# Results are keyed by their inputs, so widget interactions re-use them.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading corpus…")
def load_corpora():
    return CorpusProcessor.load_clean(), CorpusProcessor.load_annotated()


@st.cache_data(show_spinner=False)
def get_corpus_stats():
    return CorpusProcessor.corpus_stats(load_corpora()[0])


@st.cache_data(show_spinner=False, max_entries=32)
def run_xray(keyword: str, whole_word: bool, max_sentences: int):
    return SyntacticXRay().analyse(keyword, max_sentences=max_sentences,
                                   corpus=load_corpora()[0], whole_word=whole_word)


@st.cache_data(show_spinner=False, max_entries=16)
def run_spider(keyword: str, whole_word: bool, max_emails: int):
    return EntitySpiderweb().analyse(keyword, max_emails=max_emails,
                                     corpus=load_corpora()[0], whole_word=whole_word)


@st.cache_data(show_spinner=False, max_entries=16)
def run_lda(keyword: str, whole_word: bool, num_topics: int, interpret: bool):
    return ContextDecoder().analyse(keyword, num_topics=num_topics, interpret_topics=interpret,
                                    corpus=load_corpora()[0], whole_word=whole_word)


@st.cache_data(show_spinner=False, max_entries=32)
def run_timeline(keyword: str, whole_word: bool, order: str):
    return DiachronicTimeline().analyse(keyword, annotated_corpus=load_corpora()[1],
                                        whole_word=whole_word, order=order)


@st.cache_data(show_spinner=False, max_entries=256)
def assess_single(body: str, labels: tuple):
    return ThreatAssessor().assess_email(body, list(labels))


@st.cache_data(show_spinner=False)
def get_ai_wordlists():
    """Vocabulary-wide, so computed once per process — not once per keyword."""
    return AIWordlistExpander.get_dynamic_wordlists(load_corpora()[0])

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
QUICK_TARGETS = ["pizza", "massage", "island", "model", "party", "travel"]
st.session_state.setdefault("keyword", "pizza")


def _set_keyword(word: str) -> None:
    st.session_state["keyword"] = word


with st.sidebar:
    render_html("""
    <div style="padding:18px 0 10px 0">
        <div style="font-size:1.5rem;font-weight:800;color:#63b3ed;letter-spacing:-0.03em">🔬 FEAP</div>
        <div style="font-size:0.72rem;color:#718096;margin-top:2px;text-transform:uppercase;letter-spacing:0.08em">
            Forensic Entity-Action Profiler
        </div>
        <div style="font-size:0.68rem;color:#4a5568;margin-top:8px;border-top:1px solid rgba(99,179,237,0.12);padding-top:8px;">
            EFLU · Ameen K.P
        </div>
    </div>
    """)

    st.markdown("---")
    render_html("<p class='sidebar-label'>Target Keyword</p>")
    st.text_input(
        "Enter keyword to profile", key="keyword",
        label_visibility="collapsed", placeholder="e.g. pizza, massage, island…",
    )
    whole_word = st.toggle(
        "Whole-word match", value=True, key="whole_word",
        help="On: matches the word and simple inflections (model → models, modeling) "
             "but not substrings (pie ✗ piece, may ✗ mayor). Off: plain substring search.",
    )

    render_html("<p class='sidebar-label'>Quick Targets</p>")
    quick_cols = st.columns(2)
    for i, word in enumerate(QUICK_TARGETS):
        quick_cols[i % 2].button(word, key=f"quick_{word}", on_click=_set_keyword, args=(word,),
                                  width="stretch")

    st.markdown("---")

    try:
        stats = get_corpus_stats()
        render_html("<p class='sidebar-label'>Corpus Status</p>")
        render_html(f"""
        <div style="font-size:0.8rem;color:#a0aec0;line-height:2.1">
            📧 <b style="color:#63b3ed">{stats['total_emails']:,}</b> emails loaded<br>
            🔤 <b style="color:#63b3ed">{stats['total_tokens']:,}</b> tokens<br>
            👤 <b style="color:#63b3ed">{stats['unique_senders']:,}</b> unique senders<br>
            ✅ <b style="color:#68d391">READY</b>
        </div>""")
    except Exception as e:
        st.error(f"Corpus load error: {e}")

    st.markdown("---")

    render_html("<p class='sidebar-label'>Active Modules</p>")
    mod_xray     = st.toggle("🧬 Syntactic X-Ray",     value=True)
    mod_spider   = st.toggle("🕸 Entity Spiderweb",    value=True)
    mod_topic    = st.toggle("🫧 Context Decoder",     value=True)
    mod_timeline = st.toggle("📈 Diachronic Timeline", value=True)
    mod_threat   = st.toggle("⚡ Threat Assessor",     value=True)

    st.markdown("---")
    xray_sentences = st.slider("Sentences to parse (X-Ray)", 10, 100, 30, step=5)
    num_topics     = st.slider("LDA Topics (Context Decoder)", 3, 8, 5)

    use_ai_topics = st.toggle(
        "🤖 AI Topic Interpretation", value=True,
        help="Use the zero-shot model to label LDA topics with forensic descriptions "
             "(falls back to rules when the model is unavailable).")
    use_ai_wordlists = st.toggle(
        "🤖 AI Wordlist Expansion", value=False,
        help="Use the zero-shot model to expand the food-cipher and euphemism wordlists "
             "from corpus vocabulary. Runs once; the first run downloads the model.")

    render_html("<div style='font-size:0.68rem;color:#4a5568;margin-top:18px'>"
                "⚠️ AI features need <code>torch</code> + <code>transformers</code>.<br>"
                "See requirements.txt.</div>")


# ─────────────────────────────────────────────────────────────────────────────
keyword = (st.session_state.get("keyword") or "").strip() or "pizza"
kw_pat  = keyword_pattern(keyword, whole_word)
kw_slug = slug(keyword)

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
render_html(f"""
<div class="hero">
    <h1 class="hero-title">Forensic Entity-Action <span>Profiler</span></h1>
    <p class="hero-sub">
        Computational pragmatics framework for decoding weaponized language within illicit networks.<br>
        Currently profiling: <strong style="color:#3182ce">"{esc(keyword)}"</strong> across the Epstein FOIA corpus
        <span style="font-size:0.8rem">({'whole-word' if whole_word else 'substring'} match)</span>.
    </p>
</div>
""")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CORPUS
# ─────────────────────────────────────────────────────────────────────────────
try:
    all_emails, annotated_emails = load_corpora()
    cp           = CorpusProcessor()
    kw_hits      = cp.query_by_keyword(all_emails, keyword, whole_word)
    kw_sentences = cp.extract_sentences_with_keyword(kw_hits, keyword, whole_word)
except Exception as e:
    st.error(f"Engine error: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# AI WORDLISTS
# ─────────────────────────────────────────────────────────────────────────────
active_food = set(FOOD_CODE_WORDS)
active_euph = set(EUPHEMISM_WORDS)
if use_ai_wordlists:
    with st.spinner("🤖 Building AI-expanded forensic wordlists from corpus vocabulary…"):
        ai_wordlists = get_ai_wordlists()
    active_food = set(ai_wordlists["food_code"])
    active_euph = set(ai_wordlists["euphemism"])
    new_food = sorted(active_food - FOOD_CODE_WORDS)
    new_euph = sorted(active_euph - EUPHEMISM_WORDS)
    with st.expander(f"🤖 AI wordlists: +{len(new_food)} food-cipher terms, "
                     f"+{len(new_euph)} euphemism terms"):
        if new_food or new_euph:
            render_html(
                "<b>Food-cipher additions:</b> "
                + ("".join(f'<span class="wl-tag">{esc(w)}</span>' for w in new_food) or "—")
                + "<br><b>Euphemism additions:</b> "
                + ("".join(f'<span class="wl-tag">{esc(w)}</span>' for w in new_euph) or "—"))
        else:
            st.caption("No additions — the model is unavailable or found no new terms; "
                       "the static seed lists are in use.")

# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW METRICS
# ─────────────────────────────────────────────────────────────────────────────
oc1, oc2, oc3, oc4 = st.columns(4)
metric_card(f"{len(kw_hits):,}", f'Emails containing "{keyword}"', oc1)
metric_card(f"{len(kw_sentences):,}", "Sentences with keyword", oc2)
metric_card(f"{len(kw_hits) / max(len(all_emails), 1) * 100:.1f}%", "Corpus coverage", oc3)
hedged = sum(1 for s in kw_sentences if has_hedge(s["sentence"]))
metric_card(f"{hedged / max(len(kw_sentences), 1) * 100:.0f}%", "Sentences with modal hedging", oc4)

if not kw_hits:
    st.info(f'No emails contain "{keyword}". Try another keyword'
            + (" or switch off whole-word matching." if whole_word else "."))

render_html("<div style='margin:8px 0'></div>")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_labels = ["🧬 X-Ray", "🕸 Spiderweb", "🫧 Topic Decoder",
              "📈 Timeline", "⚡ Threat Assessor", "📋 Raw Corpus"]
tabs = st.tabs(tab_labels)
kw_html = f'<span class="kw-ref">"{esc(keyword)}"</span>'


# ════════════════════════════════════════════════════════════
# MODULE 1 — SYNTACTIC X-RAY
# ════════════════════════════════════════════════════════════
with tabs[0]:
    section_header("🧬", "Syntactic X-Ray", "Dependency Parsing · spaCy")
    render_html(f"""<p class="module-intro">
    Dependency-parse sentences containing {kw_html} to reveal whether it’s surrounded by
    logistical verbs, modal hedges, or subordinate clauses — the hallmarks of a pragmatic cipher.
    </p>""")

    if not mod_xray:
        st.info("Module disabled in sidebar.")
    else:
        try:
            with st.spinner("Parsing sentences with spaCy…"):
                xdata = run_xray(keyword, whole_word, xray_sentences)

            x1, x2, x3, x4 = st.columns(4)
            metric_card(xdata["total_emails_with_keyword"], "Emails hit", x1)
            metric_card(xdata["sentences_analysed"], "Sentences parsed", x2)
            metric_card(f'{xdata["hedge_percentage"]}%', "Modal hedging", x3)
            metric_card(f'{xdata["logistic_verb_percentage"]}%', "Logistic root verbs", x4)

            render_html("<div style='margin:12px 0'></div>")
            col_l, col_r = st.columns(2)
            n_parsed = max(xdata["sentences_analysed"], 1)

            with col_l:
                render_html("<p class='panel-title'>🔴 Top Root Verbs <span style='font-weight:400'>(🚨 = logistic)</span></p>")
                for verb, count in xdata["top_root_verbs"]:
                    pct = round(count / n_parsed * 100, 1)
                    is_logistic = verb.lower() in LOGISTIC_VERBS
                    render_html(score_bar(f"{'🚨' if is_logistic else '▸'} {verb}", pct,
                                          "#fc8181" if is_logistic else "#63b3ed"))

            with col_r:
                render_html("<p class='panel-title'>🔵 Top Modifiers on Keyword</p>")
                if not xdata["top_modifiers"]:
                    st.caption("No content-word modifiers attach directly to the keyword.")
                for mod, count in xdata["top_modifiers"][:8]:
                    pct = round(count / n_parsed * 100, 1)
                    render_html(score_bar(f"▸ {mod}", pct, "#d6bcfa"))

            render_html("<div style='margin:18px 0 10px 0'></div>")
            section_header("📜", "Sentence Dependency Viewer", "Interactive")

            details = xdata["sentence_details"]
            if details:
                sent_opts = [
                    f'[{i+1}] {"🚨 " if d["is_logistic"] else ""}{"〰 " if d["hedges"] else ""}'
                    f'{d["sender"][:25]} — {d["subject"][:40]}'
                    for i, d in enumerate(details)
                ]
                chosen_idx = st.selectbox("Select sentence to inspect:", range(len(sent_opts)),
                                          format_func=lambda i: sent_opts[i])
                detail = details[chosen_idx]

                hedge_tags = "".join(f'<span class="wl-tag hedge">〰 {esc(h)}</span>' for h in detail["hedges"])
                render_html(f"""
                <div class="email-card">
                    <div class="meta">{email_meta({"id": detail["email_id"], "sender": detail["sender"], "subject": detail["subject"]})}</div>
                    <div class="body-snippet" style="margin-top:8px">{highlight_keyword(detail["sentence"], kw_pat)}</div>
                    <div style="margin-top:6px">{hedge_tags}</div>
                </div>""")

                render_email_overlay(cp.get_email_by_id(detail["email_id"]), kw_pat)

                render_html("<p style='color:#718096;font-size:0.78rem;margin:10px 0 5px'>Token / POS tags:</p>")
                pills_html = "".join(pos_pill(t["text"], t["pos"], t["is_keyword"]) for t in detail["tokens"])
                render_html(f'<div style="line-height:2.2">{pills_html}</div>')

                if detail["edges"]:
                    import pandas as pd  # type: ignore
                    render_html("<p style='color:#718096;font-size:0.78rem;margin:14px 0 5px'>Dependency edges:</p>")
                    st.dataframe(
                        pd.DataFrame(detail["edges"]).rename(
                            columns={"from": "Head", "to": "Dependent", "dep": "Relation"}),
                        hide_index=True,
                    )

                csv_download(
                    [{"email_id": d["email_id"], "sender": d["sender"], "subject": d["subject"],
                      "sentence": d["sentence"], "root": d["root"],
                      "logistic_root": d["is_logistic"], "hedges": ", ".join(d["hedges"])}
                     for d in details],
                    f"feap_xray_{kw_slug}.csv", "⬇️ Download parsed sentences (CSV)", key="dl_xray")
            else:
                st.warning(f"No sentences parsed for '{keyword}'. Try a different keyword.")

        except RuntimeError as e:
            st.warning(f"⚠️ {e}")
            st.markdown("**Fallback — Keyword Concordance (no spaCy required):**")
            for row in kw_sentences[:15]:
                email_card({"id": row["id"], "sender": row["sender"], "subject": row["subject"],
                            "body": row["sentence"]}, kw_pat, snippet_len=600)
                render_email_overlay(cp.get_email_by_id(row["id"]), kw_pat)

# ════════════════════════════════════════════════════════════
# MODULE 2 — ENTITY SPIDERWEB
# ════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("🕸", "Entity Spiderweb", "spaCy NER · NetworkX · Co-occurrence")
    render_html(f"""<p class="module-intro">
    spaCy Named Entity Recognition extracts <b>PERSON</b>, <b>ORGANIZATION</b>, <b>LOCATION</b> and
    <b>FACILITY</b> mentions from every email containing {kw_html}. Edge weights count how often two
    entities appear in the same email; <b>brokers</b> are entities that bridge otherwise separate clusters
    (betweenness centrality).
    <br><span style="color:#d69e2e;font-size:0.78rem">💡 Click a node in the graph (or pick one from the list) to see the emails behind it.</span>
    </p>""")

    if not mod_spider:
        st.info("Module disabled in sidebar.")
    else:
        spider_col1, spider_col2 = st.columns([3, 1])
        with spider_col1:
            spider_emails = st.slider("Max emails to process (NER is slow):", 20, 300, 80, key="spider_n")
        with spider_col2:
            render_html("<div style='height:28px'></div>")
            run_spider_clicked = st.button("⚙️ Build Entity Graph", key="btn_spider")

        spider_key = (keyword, whole_word)
        if run_spider_clicked:
            st.session_state["spider_key"] = spider_key

        if st.session_state.get("spider_key") == spider_key:
            try:
                with st.spinner(f"Running NER on up to {spider_emails} emails…"):
                    sdata = run_spider(keyword, whole_word, spider_emails)

                s1, s2, s3, s4 = st.columns(4)
                metric_card(sdata["total_emails"],   "Emails processed",     s1)
                metric_card(sdata["total_entities"], "Named entities found", s2)
                metric_card(len(sdata["nodes"]),     "Entities in graph",    s3)
                metric_card(len(sdata["edges"]),     "Co-occurrence edges",  s4)

                if not sdata["nodes"]:
                    st.warning("No named entities extracted. Try a broader keyword.")
                else:
                    PLACEHOLDER = "— pick an entity —"
                    entity_names = [n["id"] for n in sdata["nodes"]]
                    entity_opts = [PLACEHOLDER] + entity_names
                    if st.session_state.get("spider_entity_sel") not in entity_opts:
                        st.session_state["spider_entity_sel"] = PLACEHOLDER

                    col_l, col_r = st.columns([2, 1])

                    with col_l:
                        try:
                            from streamlit_agraph import agraph, Node, Edge, Config  # type: ignore
                            max_mentions = max(n["mentions"] for n in sdata["nodes"])
                            nodes = [Node(
                                id=n["id"], label=n["id"][:18],
                                size=12 + 18 * n["mentions"] / max(max_mentions, 1),
                                color=ENTITY_COLOURS.get(n["type"], "#a0aec0"),
                                title=f'{n["id"]} · {n["type"]} · {n["mentions"]} emails',
                                font={"color": "#1e293b", "size": 11, "face": "Inter"},
                            ) for n in sdata["nodes"]]
                            edges = [Edge(
                                source=e["source"], target=e["target"],
                                width=min(e["weight"] * 0.8 + 1, 6),
                                color="rgba(99,179,237,0.35)",
                                title=f'{e["weight"]} shared emails',
                            ) for e in sdata["edges"]]
                            config = Config(
                                width="100%", height=500,
                                directed=False, physics=True,
                                backgroundColor="#ffffff",
                                nodeHighlightBehavior=True,
                                highlightColor="#3182ce",
                            )
                            clicked = agraph(nodes=nodes, edges=edges, config=config)
                            # A new click overrides the dropdown; an unchanged
                            # click value must not undo a later manual pick.
                            if (clicked and clicked in entity_names
                                    and clicked != st.session_state.get("_spider_last_click")):
                                st.session_state["_spider_last_click"] = clicked
                                st.session_state["spider_entity_sel"] = clicked

                        except ImportError:
                            import pandas as pd  # type: ignore
                            st.info("💡 Install `streamlit-agraph` for the interactive graph: `pip install streamlit-agraph`")
                            if sdata["edges"]:
                                st.dataframe(pd.DataFrame(sdata["edges"][:30]).rename(columns={
                                    "source": "Entity A", "target": "Entity B",
                                    "weight": "Co-occurrence Weight"}), hide_index=True)

                        render_html("""
                        <div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;font-size:0.75rem">
                            <span>🔵 <span style="color:#3182ce">PERSON</span></span>
                            <span>🔴 <span style="color:#e53e3e">ORGANIZATION</span></span>
                            <span>🟢 <span style="color:#38a169">GPE / LOCATION</span></span>
                            <span>🟡 <span style="color:#b7791f">FACILITY</span></span>
                            <span style="color:#718096">· node size = number of emails</span>
                        </div>""")

                    with col_r:
                        render_html("<p class='panel-title'>🏆 Top Connected Entities</p>")
                        max_weight = sdata["top_entities"][0][1] if sdata["top_entities"] else 1
                        for name, weight in sdata["top_entities"][:10]:
                            render_html(score_bar(name[:30], round(weight / max(max_weight, 1) * 100, 1),
                                                  "#68d391", display=f"{weight} links"))

                        if sdata.get("top_brokers"):
                            render_html("<p class='panel-title' style='margin-top:14px'>🌉 Top Brokers</p>")
                            max_b = sdata["top_brokers"][0][1] or 1
                            for name, b in sdata["top_brokers"][:5]:
                                render_html(score_bar(name[:30], round(b / max_b * 100, 1), "#b794f4",
                                                      display=f"{b:.2f}"))

                        render_html("<p class='panel-title' style='margin-top:14px'>🔍 Explore an entity</p>")
                        selected_entity = st.selectbox("Select entity:", entity_opts, key="spider_entity_sel")
                        st.button("✖ Clear selection", key="spider_clear",
                                  on_click=lambda: st.session_state.update(spider_entity_sel=PLACEHOLDER))

                    # ── Selected entity → email list ──────────────────────────
                    if selected_entity and selected_entity != PLACEHOLDER:
                        node_emails = sdata["node_email_map"].get(selected_entity, [])
                        st.markdown("---")
                        render_html(f"""
                        <div class="node-hint">
                            🕸 Showing <strong>{len(node_emails)}</strong> emails mentioning entity:
                            <strong style="color:#3182ce">{esc(selected_entity)}</strong>
                        </div>""")

                        for nem in node_emails[:10]:
                            email_card(nem, kw_pat, snippet_len=300)
                            render_email_overlay(nem, kw_pat)
                        if len(node_emails) > 10:
                            st.caption(f"…and {len(node_emails) - 10} more. Use the Raw Corpus tab to browse them all.")

                    csv_download(sdata["nodes"], f"feap_entities_{kw_slug}.csv",
                                 "⬇️ Download entities (CSV)", key="dl_nodes")
                    csv_download(sdata["edges"], f"feap_edges_{kw_slug}.csv",
                                 "⬇️ Download co-occurrence edges (CSV)", key="dl_edges")

            except RuntimeError as e:
                st.warning(f"⚠️ {e}")
        else:
            render_html("""
            <div class="email-card empty-state">
                Click <strong style="color:#63b3ed">Build Entity Graph</strong> to run NER analysis.<br>
                <span style="font-size:0.78rem">Processing time depends on the number of emails selected.</span>
            </div>""")

# ════════════════════════════════════════════════════════════
# MODULE 3 — CONTEXT DECODER (LDA)
# ════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("🫧", "Hidden Context Decoder", "Unsupervised LDA · Gensim")
    ai_note = ("<br><span style='color:#63b3ed;font-size:0.78rem'>🤖 AI topic interpretation enabled — "
               "each topic will receive a forensic description.</span>") if use_ai_topics else ""
    render_html(f"""<p class="module-intro">
    Latent Dirichlet Allocation clusters the <em>surrounding</em> vocabulary of emails containing
    {kw_html} into hidden semantic groups. The keyword itself and e-mail boilerplate (headers,
    dates, “Sent from my iPhone”) are excluded so topics reflect what the messages are about.
    {ai_note}
    </p>""")

    if not mod_topic:
        st.info("Module disabled in sidebar.")
    else:
        run_lda_clicked = st.button("⚙️ Run LDA Topic Modeling", key="btn_lda")
        lda_key = (keyword, whole_word)
        if run_lda_clicked:
            st.session_state["lda_key"] = lda_key

        if st.session_state.get("lda_key") == lda_key:
            try:
                spinner_text = ("Running LDA and interpreting topics (first AI run downloads the model)…"
                                if use_ai_topics else "Running LDA passes…")
                with st.spinner(spinner_text):
                    tdata = run_lda(keyword, whole_word, num_topics, use_ai_topics)

                if "error" in tdata:
                    st.error(tdata["error"])
                else:
                    t1, t2 = st.columns(2)
                    metric_card(tdata["emails_analysed"], "Emails in LDA corpus", t1)
                    metric_card(tdata["num_topics"],      "Hidden topics extracted", t2)

                    render_html("<div style='margin:14px 0'></div>")

                    dom_counts = Counter(d["dominant_topic"] for d in tdata["doc_topics"])
                    cols = st.columns(min(num_topics, 3))
                    for i, topic in enumerate(tdata["topics"]):
                        color = TOPIC_COLOURS[i % len(TOPIC_COLOURS)]
                        max_w = max(topic["weights"] or [1])
                        words_html = "".join(
                            f'<span style="display:inline-block;margin:3px;padding:4px 10px;border-radius:8px;'
                            f'font-size:0.75rem;background:{color}1a;color:#2d3748;'
                            f'border:1px solid {color}66;opacity:{0.55 + 0.45 * (w / max_w):.2f}">{esc(word)}</span>'
                            for word, w in zip(topic["words"], topic["weights"])
                        )
                        interp = topic.get("interpretation")
                        ai_html = ""
                        if interp:
                            ai_badge = ('<span class="ai-badge">🤖 AI</span>' if interp.get("ai_powered")
                                        else '<span class="ai-badge ai-badge-fallback">RULE</span>')
                            ai_html = f"""
<div class="ai-interp-card">
    {ai_badge}
    <span style="font-weight:700;color:#2d3748;margin-left:6px">{esc(interp.get('best_label', ''))}</span><br>
    <span style="color:#4a5568;font-size:0.8rem">{esc(interp.get('description', ''))}</span><br>
    <span style="font-size:0.78rem;margin-top:4px;display:block">{esc(interp.get('forensic_significance', ''))}</span>
</div>"""
                        render_html(f"""
<div class="email-card" style="border-color:{color}66;text-align:center">
    <div style="color:{color};font-weight:700;font-size:0.9rem;margin-bottom:8px">
        ◉ {esc(topic["label"])} <span style="color:#718096;font-weight:500;font-size:0.75rem">· {dom_counts.get(topic["id"], 0)} emails</span>
    </div>
    <div style="line-height:2.2">{words_html}</div>
    {ai_html}
</div>""", cols[i % len(cols)])

                    if tdata["doc_topics"]:
                        import pandas as pd  # type: ignore
                        df_dist = pd.DataFrame([
                            {"Topic": f"Topic {k + 1}", "Emails": dom_counts.get(k, 0)}
                            for k in range(tdata["num_topics"])
                        ])
                        render_html("<div style='margin:18px 0 8px 0'></div>")
                        section_header("📊", "Topic Document Distribution")
                        st.bar_chart(df_dist.set_index("Topic"), color="#63b3ed")

                        section_header("🔎", "Most Representative Emails")
                        topic_choice = st.selectbox(
                            "Topic:", range(tdata["num_topics"]),
                            format_func=lambda k: f"Topic {k + 1} — " + ", ".join(tdata["topics"][k]["words"][:4]),
                            key="lda_topic_sel",
                        )
                        rep = sorted((d for d in tdata["doc_topics"] if d["dominant_topic"] == topic_choice),
                                     key=lambda d: d["confidence"], reverse=True)[:5]
                        if not rep:
                            st.caption("No email has this as its dominant topic.")
                        for d in rep:
                            em = cp.get_email_by_id(d["email_id"])
                            if em:
                                email_card(em, kw_pat, snippet_len=300,
                                           extra_html=f'<span class="wl-tag">{d["confidence"]}% topic share</span>',
                                           full_body_inline=True)

                        csv_download(tdata["doc_topics"], f"feap_topics_{kw_slug}.csv",
                                     "⬇️ Download email → topic assignments (CSV)", key="dl_topics")

            except RuntimeError as e:
                st.warning(f"⚠️ {e}")
                st.markdown("**Fallback — Top contextual words (no Gensim):**")
                stop = ContextDecoder._STOP | ContextDecoder._keyword_stopwords(keyword)
                counts = Counter(t for e in kw_hits for t in tokenize_words(e.get("body", ""))
                                 if len(t) >= 4 and t not in stop)
                pills = "".join(
                    f'<span class="wl-tag">{esc(w)} <span style="color:#718096">×{c}</span></span>'
                    for w, c in counts.most_common(20)
                )
                render_html(f'<div style="margin-top:8px">{pills}</div>')
        else:
            render_html("""
            <div class="email-card empty-state">
                Click <strong style="color:#63b3ed">Run LDA Topic Modeling</strong> to decode hidden contexts.<br>
                <span style="font-size:0.78rem">Requires Gensim. Adjust topic count in the sidebar.</span>
            </div>""")

# ════════════════════════════════════════════════════════════
# MODULE 4 — DIACHRONIC TIMELINE
# ════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("📈", "Diachronic Timeline", "Syntactic Shift · Pandas")
    render_html(f"""<p class="module-intro">
    Track how <strong>modal hedging density</strong> and <strong>logistical verb density</strong>
    evolve over time across emails containing {kw_html}. The corpus has no date field, so each email's
    date is recovered from its reply/forward headers (“On Apr 6, 2016 … wrote”). Hedges are matched as
    whole words and phrases (“I think”, “sort of”), logistic verbs in any inflection (arranged, sent,
    picking up). Spikes in hedging can indicate periods of external pressure or legal scrutiny.
    </p>""")

    if not mod_timeline:
        st.info("Module disabled in sidebar.")
    else:
        try:
            order_choice = st.radio(
                "Order emails by:", ["Estimated date", "Corpus order"], horizontal=True, key="tl_order",
                help="Estimated date keeps only emails whose date could be recovered. "
                     "Corpus order keeps every matching email.")
            tlinedata = run_timeline(keyword, whole_word, "date" if order_choice == "Estimated date" else "corpus")

            if "error" in tlinedata:
                st.warning(tlinedata["error"])
            else:
                import pandas as pd  # type: ignore

                by_date = tlinedata["order"] == "date"
                if by_date:
                    start, end = tlinedata["date_range"]
                    st.caption(f"📅 {tlinedata['dated_emails']} of {tlinedata['matched_emails']} matching emails "
                               f"carry a recoverable date · {start} → {end}")
                elif order_choice == "Estimated date":
                    st.caption(f"📅 Only {tlinedata['dated_emails']} matching emails carry a recoverable date — "
                               "showing corpus order instead.")

                t1, t2, t3, t4 = st.columns(4)
                metric_card(tlinedata["total_emails"],               "Emails in timeline",   t1)
                metric_card(f'{tlinedata["avg_hedge_density"]}%',    "Avg hedge density",    t2)
                metric_card(f'{tlinedata["avg_logistic_density"]}%', "Avg logistic density", t3)
                peak = tlinedata["peak_hedge_email"]
                metric_card(f'{peak["hedge_density"]}%', "Peak hedge density", t4)

                df = pd.DataFrame(tlinedata["timeline"])
                window = tlinedata.get("window", 10)
                x_col, x_label = ("date", "Estimated date") if by_date else ("index", "Email # (corpus order)")

                render_html("<div style='margin:14px 0'></div>")
                section_header("📊", "Hedge & Logistic Density Over Time" if by_date
                               else "Hedge & Logistic Density Over Email Sequence")

                chart_col, info_col = st.columns([3, 1])
                with chart_col:
                    st.line_chart(
                        df.set_index(x_col)[["hedge_rolling", "logistic_rolling"]].rename(
                            columns={"hedge_rolling": "Modal hedging",
                                     "logistic_rolling": "Logistic verbs"}),
                        color=["#fc8181", "#63b3ed"],
                        x_label=x_label, y_label="% of tokens",
                    )

                def _peak_html(title: str, rec: dict, field: str, label: str, colour: str) -> str:
                    when = rec.get("date")
                    when_html = (f"Date: {esc(pd.Timestamp(when).date().isoformat())}<br>"
                                 if when is not None and not pd.isna(when) else "")
                    return f"""
                    <p class='panel-title' style='font-size:0.78rem;margin-top:10px'>{title}</p>
                    <div style="font-size:0.75rem;color:#718096;line-height:1.7">
                        ID: <span class="email-id">{esc(rec.get("email_id", "?"))}</span><br>
                        {when_html}
                        Sender: {esc(str(rec.get("sender", "?"))[:24])}<br>
                        {label}: <span style="color:{colour}">{rec[field]}%</span>
                    </div>"""

                with info_col:
                    render_html(f"""
                    <p class='panel-title'>Legend</p>
                    <div style="font-size:0.78rem;line-height:2">
                        <span style="color:#fc8181">—</span> Modal hedging ({window}-email rolling avg)<br>
                        <span style="color:#63b3ed">—</span> Logistic verbs ({window}-email rolling avg)
                    </div>
                    {_peak_html("Peak Hedging Email", peak, "hedge_density", "Hedge", "#e53e3e")}
                    {_peak_html("Peak Logistic Email", tlinedata["peak_logistic_email"], "logistic_density", "Logistic", "#3182ce")}
                    """)

                render_html("<div style='margin:18px 0 8px 0'></div>")
                if by_date:
                    section_header("📌", "Flagged Term Density", "Annotated cipher/euphemism terms per month")
                    monthly = df.set_index("date").resample("MS")["flag_count"].sum()
                    monthly.index = monthly.index.strftime("%Y-%m")
                    st.bar_chart(monthly, color="#d6bcfa", x_label="Month", y_label="Flagged terms")
                else:
                    section_header("📌", "Flagged Term Density", "Annotated cipher/euphemism terms per email")
                    st.bar_chart(df.set_index("index")["flag_count"], color="#d6bcfa",
                                 x_label=x_label, y_label="Flagged terms")

                show_cols = ["email_id", "date", "sender", "subject", "hedge_density", "logistic_density",
                             "hedge_rolling", "logistic_rolling", "flag_count"]
                table = df[show_cols].copy()
                table["date"] = table["date"].dt.strftime("%Y-%m-%d")
                with st.expander("📄 View raw timeline data"):
                    st.dataframe(
                        table.rename(columns={
                            "email_id": "Email ID", "date": "Est. date", "hedge_density": "Hedge %",
                            "logistic_density": "Logistic %", "hedge_rolling": "Hedge Rolling",
                            "logistic_rolling": "Logistic Rolling", "flag_count": "Flags",
                        }),
                        hide_index=True,
                    )
                csv_download(table, f"feap_timeline_{kw_slug}.csv",
                             "⬇️ Download timeline (CSV)", key="dl_timeline")

                render_html("<p class='panel-title' style='margin-top:16px'>📧 Open email by ID</p>")
                tl_opts = ["— pick —"] + df["email_id"].tolist()
                if st.session_state.get("tl_email_sel") not in tl_opts:
                    st.session_state["tl_email_sel"] = "— pick —"
                selected_tl_id = st.selectbox("Select email ID:", tl_opts, key="tl_email_sel")
                if selected_tl_id != "— pick —":
                    tl_email = cp.get_email_by_id(selected_tl_id)
                    if tl_email:
                        row = table[table["email_id"] == selected_tl_id].iloc[0]
                        date_tag = (f'<span class="wl-tag">📅 {esc(row["date"])}</span>'
                                    if isinstance(row["date"], str) else "")
                        email_card(tl_email, kw_pat, snippet_len=300, extra_html=(
                            date_tag
                            + f'<span class="wl-tag hedge">Hedge {row["hedge_density"]}%</span>'
                            + f'<span class="wl-tag">Logistic {row["logistic_density"]}%</span>'
                            + f'<span class="wl-tag">Flags {int(row["flag_count"])}</span>'))
                        render_email_overlay(tl_email, kw_pat)
                    else:
                        st.warning("Email not found in corpus.")

        except RuntimeError as e:
            st.warning(f"⚠️ {e}")

# ════════════════════════════════════════════════════════════
# MODULE 5 — THREAT ASSESSOR
# ════════════════════════════════════════════════════════════
def render_scores(res: dict) -> str:
    return "".join(score_bar(s["label"], s["score"], label_colour(s["label"], i))
                   for i, s in enumerate(res["all_scores"]))


def rule_based_signals(body: str) -> list:
    """Heuristic intent signals used when the transformer is unavailable."""
    signals = []
    if count_logistic_verbs(body):
        signals.append("🚨 Logistical Coordination")
    hedges = find_hedges(body)
    if hedges:
        signals.append(f"🔶 Modal Hedging ({', '.join(sorted(set(hedges))[:3])})")
    food_hits = sorted({m.group(0).lower() for m in food_re.finditer(body)})
    if food_hits:
        signals.append(f"🟡 Food Code Word ({', '.join(food_hits[:3])})")
    euph_hits = sorted({m.group(0).lower() for m in euph_re.finditer(body)})
    if euph_hits:
        signals.append(f"🔴 Euphemism ({', '.join(euph_hits[:3])})")
    return signals


food_re = words_regex(active_food)
euph_re = words_regex(active_euph)

with tabs[4]:
    section_header("⚡", "Threat Assessor", "Zero-Shot Classification · HuggingFace")
    render_html("""<p class="module-intro">
    A HuggingFace Transformer (zero-shot NLI) scores each email’s pragmatic intent against the
    forensic labels below. Scores are relative across the labels you provide (they sum to 100%),
    so they indicate the <em>closest</em> label rather than an absolute probability of wrongdoing.
    </p>""")

    if not mod_threat:
        st.info("Module disabled in sidebar.")
    else:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            render_html("<p style='color:#a0aec0;font-size:0.8rem;margin-bottom:6px'>Custom forensic labels (one per line):</p>")
            labels_input = st.text_area(
                "Labels", value="\n".join(FORENSIC_LABELS),
                height=160, label_visibility="collapsed", key="labels_ta",
            )
        with col_b:
            render_html("<div style='height:28px'></div>")
            single_mode = st.toggle("Single-email mode", value=False)
            n_assess    = st.slider("Emails to assess (batch):", 3, 50, 8, key="n_assess")

        custom_labels = list(dict.fromkeys(l.strip() for l in labels_input.splitlines() if l.strip()))
        if len(custom_labels) < 2:
            st.warning("Enter at least two distinct labels — zero-shot scores are relative between labels.")

        if not kw_hits:
            st.warning(f"No emails found for '{keyword}'.")

        elif single_mode:
            email_opts = [
                f'{e["id"][:15]} — {e["sender"][:25]} — {e["subject"][:35]}'
                for e in kw_hits[:50]
            ]
            chosen = st.selectbox("Choose email:", range(len(email_opts)),
                                  format_func=lambda i: email_opts[i])
            target_email = kw_hits[chosen]
            email_card(target_email, kw_pat, snippet_len=500, full_body_inline=True)

            if st.button("⚡ Assess This Email", key="btn_assess_single", disabled=len(custom_labels) < 2):
                try:
                    with st.spinner("Classifying (the first run loads the model, ~30s)…"):
                        result = assess_single(target_email["body"], tuple(custom_labels))
                    render_html(f"""
                    <div class="email-card" style="border-color:var(--accent)">
                        <div style="margin-bottom:12px">{threat_badge(result['top_label'], result['top_score'])}</div>
                        <strong style="font-size:0.85rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em">All Threat Scores</strong>
                        <div style="margin-top:8px">{render_scores(result)}</div>
                    </div>""")
                except (RuntimeError, ValueError) as e:
                    st.warning(f"⚠️ {e}")
                    signals = rule_based_signals(target_email["body"])
                    render_html("**Fallback — Rule-based intent signals:** " + (
                        " ".join(f'<span class="wl-tag">{esc(s)}</span>' for s in signals)
                        or "<span style='color:#718096'>No strong signals</span>"))

        else:
            # ── Batch mode ────────────────────────────────────────────────────
            batch_key = (keyword, whole_word, n_assess, tuple(custom_labels))
            batch_target = kw_hits[:n_assess]

            if st.button(f"⚡ Assess Top {len(batch_target)} Emails", key="btn_assess_batch",
                         disabled=len(custom_labels) < 2):
                prog = st.progress(0, text="Loading transformer model…")
                try:
                    ta = ThreatAssessor()
                    ta.get_pipeline()
                    prog.progress(5, text="Model ready. Classifying…")
                    scores = ta.assess_many(
                        [e["body"] for e in batch_target], custom_labels,
                        progress=lambda done, total: prog.progress(
                            5 + int(done / total * 95), text=f"Assessed {done}/{total} emails…"),
                    )
                    results = [{**e, **s} for e, s in zip(batch_target, scores)]
                    st.session_state["threat_batch"] = {"key": batch_key, "results": results, "fallback": False}
                except (RuntimeError, ValueError) as e:
                    st.warning(f"⚠️ {e} — showing rule-based intent signals instead.")
                    results = [{**e, "signals": rule_based_signals(e["body"])} for e in batch_target]
                    st.session_state["threat_batch"] = {"key": batch_key, "results": results, "fallback": True}
                finally:
                    prog.empty()

            stored = st.session_state.get("threat_batch")
            if stored and stored["key"] != batch_key:
                st.caption("Keyword, labels or batch size changed — click the button to re-assess.")
            elif stored:
                results = stored["results"]
                if stored["fallback"]:
                    st.markdown("**Fallback — Rule-based intent signals:**")
                    for res in results:
                        signals_html = (" ".join(f'<span class="wl-tag">{esc(s)}</span>' for s in res["signals"])
                                        or "<span style='color:#718096'>No strong signals</span>")
                        email_card(res, kw_pat, snippet_len=200, full_body_inline=True,
                                   extra_html=f'<div style="margin:6px 0">{signals_html}</div>')
                    csv_download([{"email_id": r["id"], "sender": r["sender"], "subject": r["subject"],
                                   "signals": "; ".join(r["signals"])} for r in results],
                                 f"feap_signals_{kw_slug}.csv", "⬇️ Download signals (CSV)", key="dl_signals")
                else:
                    top_counts = Counter(r["top_label"] for r in results)
                    summary = " ".join(
                        f'<span class="wl-tag" style="border-color:{label_colour(l, i)}">{esc(l)} × {c}</span>'
                        for i, (l, c) in enumerate(top_counts.most_common()))
                    render_html(f"<div style='margin:6px 0 14px'><b>Top-label summary:</b> {summary}</div>")

                    for res in sorted(results, key=lambda r: (r["top_label"] in BENIGN_LABELS, -r["top_score"])):
                        render_html(f"""
                        <div class="email-card">
                            <div class="meta">{email_meta(res)}</div>
                            <div style="margin:6px 0">{threat_badge(res["top_label"], res["top_score"])}</div>
                            <div class="body-snippet">{highlight_keyword(res['body'], kw_pat, 250)}</div>
                            <details class="full-email">
                                <summary>View full details &amp; scores</summary>
                                <div style="margin:12px 0 16px">
                                    <strong style="font-size:0.85rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em">All Threat Scores</strong>
                                    <div style="margin-top:8px">{render_scores(res)}</div>
                                </div>
                                <strong style="font-size:0.85rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em">Full Email Body</strong>
                                <div class="overlay-body">{highlight_keyword(res['body'], kw_pat)}</div>
                            </details>
                        </div>""")

                    csv_download(
                        [{"email_id": r["id"], "sender": r["sender"], "subject": r["subject"],
                          "top_label": r["top_label"], "top_score": r["top_score"],
                          **{f"score: {s['label']}": s["score"] for s in r["all_scores"]}}
                         for r in results],
                        f"feap_threat_{kw_slug}.csv", "⬇️ Download scores (CSV)", key="dl_threat")

# ════════════════════════════════════════════════════════════
# MODULE 6 — RAW CORPUS BROWSER
# ════════════════════════════════════════════════════════════
with tabs[5]:
    section_header("📋", "Raw Corpus Browser", f'Filtered to: "{keyword}"')
    render_html(f"""<p class="module-intro">
    Browse all <strong style="color:#3182ce">{len(kw_hits)}</strong> emails containing {kw_html}.
    Use the search to refine further.
    </p>""")

    f1, f2 = st.columns([3, 1])
    with f1:
        sub_search = st.text_input(
            "🔎 Filter further by secondary term:",
            placeholder="e.g. 'schedule', 'flight'…", key="sub_search",
        )
    with f2:
        sender_opts = ["All senders"] + [s for s, _ in Counter(e.get("sender") or "Unknown" for e in kw_hits).most_common()]
        if st.session_state.get("raw_sender") not in sender_opts:
            st.session_state["raw_sender"] = "All senders"
        sender_filter = st.selectbox("Sender:", sender_opts, key="raw_sender")

    show_emails = kw_hits
    if sender_filter != "All senders":
        show_emails = [e for e in show_emails if (e.get("sender") or "Unknown") == sender_filter]
    if sub_search.strip():
        sub_pat = keyword_pattern(sub_search, whole_word)
        show_emails = [e for e in show_emails
                       if sub_pat.search(e.get("body", "")) or sub_pat.search(e.get("subject", ""))]
    if sub_search.strip() or sender_filter != "All senders":
        render_html(f"<p style='color:#718096;font-size:0.78rem'>Showing {len(show_emails)} matching emails</p>")

    page_size   = 20
    total_pages = max(1, (len(show_emails) + page_size - 1) // page_size)
    if st.session_state.get("corpus_pg", 1) > total_pages:
        st.session_state["corpus_pg"] = 1
    page = st.slider("Page:", 1, total_pages, key="corpus_pg") if total_pages > 1 else 1
    page_emails = show_emails[(page - 1) * page_size: page * page_size]

    for email in page_emails:
        email_card(email, kw_pat, snippet_len=350)
        render_email_overlay(email, kw_pat)

    if page_emails:
        csv_download([{k: e.get(k, "") for k in ("id", "sender", "subject", "body")} for e in show_emails],
                     f"feap_emails_{kw_slug}.csv", f"⬇️ Download all {len(show_emails)} emails (CSV)",
                     key="dl_raw")
    else:
        render_html('<div class="email-card empty-state">No emails match this filter combination.</div>')

# ── Footer ───────────────────────────────────────────────────────────────────
render_html("""
<div style="margin-top:40px;padding:22px;border-top:1px solid rgba(99,179,237,0.12);text-align:center">
    <span style="color:#4a5568;font-size:0.75rem">
        🔬 Forensic Entity-Action Profiler &nbsp;·&nbsp;
        Ameen K.P | EFLU · Data: jmail.world FOIA corpus
    </span>
</div>
""")
