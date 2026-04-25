"""
app.py — Forensic Entity-Action Profiler
Streamlit Dashboard | Computational Pragmatics Framework
Author: Ameen K.P | EFLU

IMPROVEMENTS:
- Email overlay modal on click (FEAP, Timeline, Raw Corpus, Threat Assessor, Spiderweb)
- Progress bars for all background processes (NER, LDA, Threat Assessor)
- AI-powered LDA topic descriptions
- AI-powered dynamic wordlists
- Spiderweb reset-view button
- Spiderweb node click → email list overlay
- Larger email scale (slider up to 300)
- Timeline email ID click → overlay
- Threat assessor multi-email detail expansion
"""

import streamlit as st  # type: ignore
import sys, os, time

sys.path.insert(0, os.path.dirname(__file__))
from engine import (  # type: ignore
    CorpusProcessor, SyntacticXRay, EntitySpiderweb,
    ContextDecoder, DiachronicTimeline, ThreatAssessor,
    AIWordlistExpander, AITopicInterpreter,
    FORENSIC_LABELS, FOOD_CODE_WORDS, EUPHEMISM_WORDS,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forensic Entity-Action Profiler",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
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

.email-card{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:16px 20px;
    margin-bottom:12px;font-size:0.88rem;color:var(--text-primary);box-shadow:0 1px 3px rgba(0,0,0,0.05);transition:all 0.2s;}
.email-card:hover{border-color:var(--accent);background:#f1f5f9;}
.email-card .meta{color:var(--text-muted);font-size:0.78rem;margin-bottom:6px;}
.email-card .sender{color:var(--accent);font-weight:600;}
.email-card .subject{color:var(--accent4);font-weight:500;}
.email-card .body-snippet{color:var(--text-primary);line-height:1.6;}

/* Email overlay modal */
.overlay-backdrop{position:fixed;top:0;left:0;width:100vw;height:100vh;
    background:rgba(0,0,0,0.55);z-index:9998;backdrop-filter:blur(2px);}
.overlay-modal{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
    background:#ffffff;border-radius:16px;border:1px solid var(--border);
    box-shadow:0 25px 60px rgba(0,0,0,0.2);z-index:9999;
    width:min(820px,92vw);max-height:85vh;overflow-y:auto;padding:32px;}
.overlay-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px;}
.overlay-close{background:none;border:none;font-size:1.4rem;cursor:pointer;color:var(--text-muted);
    padding:4px 10px;border-radius:6px;transition:all 0.15s;}
.overlay-close:hover{background:#f1f5f9;color:var(--text-primary);}
.overlay-meta{font-size:0.8rem;color:var(--text-muted);line-height:2.2;margin-bottom:14px;
    padding:12px 16px;background:#f8fafc;border-radius:8px;border:1px solid var(--border);}
.overlay-body{font-size:0.9rem;line-height:1.85;color:var(--text-primary);
    white-space:pre-wrap;word-break:break-word;}

/* Token pills */
.token-pill{display:inline-block;padding:3px 12px;border-radius:6px;font-size:0.75rem;
    font-weight:600;margin:3px 4px;font-family:'JetBrains Mono',monospace;}
.pill-noun    {background:#ebf8ff;color:#3182ce;border:1px solid #bee3f8;}
.pill-verb    {background:#fff5f5;color:#e53e3e;border:1px solid #fed7d7;}
.pill-adj     {background:#f0fff4;color:#38a169;border:1px solid #c6f6d5;}
.pill-adv     {background:#faf5ff;color:#805ad5;border:1px solid #e9d8fd;}
.pill-keyword {background:#fffff0;color:#d69e2e;border:1px solid #fefcbf;font-weight:800;}
.pill-default {background:#f7fafc;color:#4a5568;border:1px solid #edf2f7;}

.score-bar-wrap{margin-bottom:12px;}
.score-label{font-size:0.85rem;font-weight:500;color:var(--text-primary);margin-bottom:4px;display:flex;justify-content:space-between;}
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

.dep-node{display:inline-block;background:#ebf8ff;border:1.5px solid #bee3f8;border-radius:10px;
    padding:8px 16px;margin:6px;font-family:'JetBrains Mono',monospace;font-size:0.82rem;
    color:var(--text-primary);font-weight:600;}
.dep-node.kw{background:#fffff0;border-color:#fefcbf;color:#d69e2e;box-shadow:0 0 10px rgba(214,158,46,0.1);}
.dep-label{font-size:0.62rem;color:var(--text-muted);display:block;}

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
    border-radius:10px;padding:14px 18px;margin-top:10px;font-size:0.82rem;line-height:1.7;}
.ai-badge{display:inline-block;background:#3182ce;color:white;font-size:0.65rem;font-weight:700;
    padding:2px 7px;border-radius:4px;letter-spacing:0.05em;margin-bottom:6px;vertical-align:middle;}
.ai-badge-fallback{background:#a0aec0;}

/* Progress status bar */
.progress-status{background:#f1f5f9;border:1px solid var(--border);border-radius:8px;
    padding:10px 16px;font-size:0.82rem;color:var(--text-muted);margin-bottom:12px;}

/* Clickable email ID links */
.email-id-link{color:var(--accent);cursor:pointer;text-decoration:underline;font-family:'JetBrains Mono',monospace;font-size:0.72rem;}

/* Node click hint */
.node-hint{background:#fffff0;border:1px solid #fefcbf;border-radius:8px;
    padding:8px 14px;font-size:0.78rem;color:#92400e;margin-bottom:10px;}

/* Wl-tag */
.wl-tag{display:inline-block;padding:3px 10px;border-radius:6px;font-size:0.75rem;
    font-weight:600;margin:3px;background:#f7fafc;color:#4a5568;border:1px solid #edf2f7;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def highlight_keyword(text: str, keyword: str, max_len: int = 400) -> str:
    snippet = text[:max_len]  # type: ignore
    for variant in [keyword, keyword.lower(), keyword.capitalize()]:
        snippet = snippet.replace(
            variant,
            f'<mark style="background:rgba(251,211,141,0.5);color:#92400e;'
            f'border-radius:3px;padding:0 3px;font-weight:600">{variant}</mark>'
        )
    return snippet + ("…" if len(text) > max_len else "")

def pos_pill(text: str, pos: str, is_keyword: bool) -> str:
    if is_keyword:        cls = "pill-keyword"
    elif pos.startswith("NN"): cls = "pill-noun"
    elif pos.startswith("VB"): cls = "pill-verb"
    elif pos.startswith("JJ"): cls = "pill-adj"
    elif pos.startswith("RB"): cls = "pill-adv"
    else:                      cls = "pill-default"
    return (f'<span class="token-pill {cls}">{text}'
            f'<span style="display:block;font-size:0.6rem;opacity:0.7">{pos}</span></span>')

def threat_badge(label: str, score: float) -> str:
    cls = "threat-high" if score > 60 else "threat-medium" if score > 35 else "threat-low"
    return f'<span class="threat-badge {cls}">⚡ {label} — {score}%</span>'

def score_bar(label: str, score: float, color: str = "#63b3ed") -> str:
    width = min(score, 100)
    return f"""
    <div class="score-bar-wrap">
        <div class="score-label"><span>{label}</span><strong>{score}%</strong></div>
        <div class="score-bar">
            <div class="score-fill" style="width:{width}%;background:{color}"></div>
        </div>
    </div>"""

def section_header(icon: str, title: str, badge: str = "") -> None:
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1.3rem">{icon}</span>
        <h2>{title}</h2>
        {badge_html}
    </div>""", unsafe_allow_html=True)

def metric_card(value, label: str, col=None) -> None:
    html = f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""
    (col or st).markdown(html, unsafe_allow_html=True)

def render_email_overlay(email: dict, keyword: str = "", key_suffix: str = "") -> None:
    """
    Renders a full-email overlay using a Streamlit expander styled as a modal.
    Call this AFTER showing an email card to give the user a 'View Full Email' option.
    """
    if not email:
        return
    eid     = email.get("id", "unknown")
    sender  = email.get("sender", "Unknown")
    subject = email.get("subject", "(no subject)")
    body    = email.get("body", "(no body)")
    date    = email.get("date", "")

    with st.expander(f"📧 Open full email — {eid[:20]}…", expanded=False):
        st.markdown(f"""
        <div class="overlay-meta">
            <strong>📌 ID:</strong> <code style="color:#81e6d9">{eid}</code><br>
            <strong>👤 From:</strong> <span style="color:#63b3ed">{sender}</span><br>
            <strong>📝 Subject:</strong> <span style="color:#805ad5">{subject}</span>
            {f'<br><strong>📅 Date:</strong> {date}' if date else ''}
        </div>""", unsafe_allow_html=True)

        if keyword:
            body_display = highlight_keyword(body, keyword, max_len=len(body))
        else:
            body_display = body.replace('\n', '<br>')

        st.markdown(f"""
        <div class="overlay-body" style="background:#f8fafc;border:1px solid var(--border);
             border-radius:10px;padding:20px;font-size:0.88rem;line-height:1.85;
             max-height:420px;overflow-y:auto;white-space:pre-wrap;word-break:break-word">
            {body_display}
        </div>""", unsafe_allow_html=True)

TOPIC_COLOURS = ["#63b3ed", "#fc8181", "#68d391", "#d6bcfa", "#fbd38d",
                 "#76e4f7", "#f6ad55", "#fc8181", "#9ae6b4", "#b794f4"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
_pending_kw = st.session_state.pop("_quick_kw", None)

with st.sidebar:
    st.markdown("""
    <div style="padding:18px 0 10px 0">
        <div style="font-size:1.5rem;font-weight:800;color:#63b3ed;letter-spacing:-0.03em">🔬 FEAP</div>
        <div style="font-size:0.72rem;color:#718096;margin-top:2px;text-transform:uppercase;letter-spacing:0.08em">
            Forensic Entity-Action Profiler
        </div>
        <div style="font-size:0.68rem;color:#4a5568;margin-top:8px;border-top:1px solid rgba(99,179,237,0.12);padding-top:8px;">
            EFLU · Ameen K.P
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#718096;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px'>Target Keyword</p>", unsafe_allow_html=True)

    keyword_input = st.text_input(
        "Enter keyword to profile",
        value=_pending_kw if _pending_kw else st.session_state.get("_last_kw", "pizza"),
        label_visibility="collapsed",
        placeholder="e.g. pizza, massage, island…"
    )

    st.markdown("<p style='color:#718096;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;margin-top:16px;margin-bottom:8px'>Quick Targets</p>", unsafe_allow_html=True)
    quick_cols = st.columns(2)
    for i, word in enumerate(["pizza","massage","island","model","party","travel"]):
        if quick_cols[i % 2].button(word, key=f"quick_{word}"):
            st.session_state["_quick_kw"] = word
            st.rerun()

    st.markdown("---")

    @st.cache_data(show_spinner=False)
    def load_global_corpus():
        return CorpusProcessor().load_clean()

    @st.cache_data(show_spinner=False)
    def load_global_annotated_corpus():
        return CorpusProcessor().load_annotated()

    try:
        corpus = load_global_corpus()
        cp     = CorpusProcessor()
        stats  = cp.corpus_stats(corpus)
        st.markdown("<p style='color:#718096;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px'>Corpus Status</p>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.8rem;color:#a0aec0;line-height:2.1">
            📧 <b style="color:#63b3ed">{stats['total_emails']:,}</b> emails loaded<br>
            🔤 <b style="color:#63b3ed">{stats['total_tokens']:,}</b> tokens<br>
            👤 <b style="color:#63b3ed">{stats['unique_senders']:,}</b> unique senders<br>
            ✅ <b style="color:#68d391">READY</b>
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Corpus load error: {e}")

    st.markdown("---")

    st.markdown("<p style='color:#718096;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px'>Active Modules</p>", unsafe_allow_html=True)
    mod_xray    = st.toggle("🧬 Syntactic X-Ray",     value=True)
    mod_spider  = st.toggle("🕸 Entity Spiderweb",    value=True)
    mod_topic   = st.toggle("🫧 Context Decoder",     value=True)
    mod_timeline= st.toggle("📈 Diachronic Timeline", value=True)
    mod_threat  = st.toggle("⚡ Threat Assessor",     value=True)

    st.markdown("---")
    num_topics = st.slider("LDA Topics (Module 3)", 3, 8, 5)
    max_emails = st.slider("Emails to profile (Mod 4)", 10, 100, 20)

    # AI features toggle
    use_ai_wordlists = st.toggle("🤖 AI Wordlist Expansion", value=True,
        help="Use AI model to dynamically expand forensic wordlists from corpus vocabulary")
    use_ai_topics    = st.toggle("🤖 AI Topic Interpretation", value=True,
        help="Use AI to interpret and label LDA topics with forensic descriptions")

    st.markdown("<div style='font-size:0.68rem;color:#4a5568;margin-top:18px'>⚠️ Modules 3 & 5 require optional deps.<br>See requirements.txt.</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
keyword = keyword_input.strip() or "pizza"
st.session_state["_last_kw"] = keyword

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <h1 class="hero-title">Forensic Entity-Action <span>Profiler</span></h1>
    <p class="hero-sub">
        Computational pragmatics framework for decoding weaponized language within illicit networks.<br>
        Currently profiling: <strong style="color:#3182ce">"{keyword}"</strong> across the Epstein FOIA corpus.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CORPUS
# ─────────────────────────────────────────────────────────────────────────────
try:
    all_emails       = load_global_corpus()
    annotated_emails = load_global_annotated_corpus()
    cp               = CorpusProcessor()
    kw_hits      = cp.query_by_keyword(all_emails, keyword)
    kw_sentences = cp.extract_sentences_with_keyword(kw_hits, keyword)
except Exception as e:
    st.error(f"Engine error: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# AI WORDLISTS (built once per keyword, cached in session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_ai_wordlists(keyword_key: str, enabled: bool) -> dict:
    if not enabled:
        return {
            "food_code": set(FOOD_CODE_WORDS),
            "euphemism": set(EUPHEMISM_WORDS),
            "logistic":  set(),
            "hedge":     set(),
        }
    try:
        cp = CorpusProcessor()
        corpus = cp.load_clean()
        return AIWordlistExpander.get_dynamic_wordlists(corpus)
    except Exception:
        return {
            "food_code": set(FOOD_CODE_WORDS),
            "euphemism": set(EUPHEMISM_WORDS),
            "logistic":  set(),
            "hedge":     set(),
        }

# Show a subtle AI status bar if enabled
if use_ai_wordlists:
    with st.spinner("🤖 Building AI-expanded forensic wordlists from corpus vocabulary…"):
        ai_wordlists = get_ai_wordlists(keyword, use_ai_wordlists)
        active_food = ai_wordlists["food_code"]
        active_euph = ai_wordlists["euphemism"]
        # Show diff vs static
        new_food = active_food - FOOD_CODE_WORDS
        new_euph = active_euph - EUPHEMISM_WORDS
        if new_food or new_euph:
            st.markdown(
                f'<div class="progress-status">🤖 AI Wordlist: added '
                f'<strong style="color:#63b3ed">{len(new_food)}</strong> food-cipher terms, '
                f'<strong style="color:#fc8181">{len(new_euph)}</strong> euphemism terms from corpus vocabulary.</div>',
                unsafe_allow_html=True
            )
else:
    active_food = set(FOOD_CODE_WORDS)
    active_euph = set(EUPHEMISM_WORDS)

# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW METRICS
# ─────────────────────────────────────────────────────────────────────────────
oc1, oc2, oc3, oc4 = st.columns(4)
metric_card(f"{len(kw_hits):,}", f'Emails containing "{keyword}"', oc1)
metric_card(f"{len(kw_sentences):,}", "Sentences with keyword", oc2)
metric_card(f"{round(len(kw_hits)/max(len(all_emails),1)*100,1)}%", "Corpus coverage", oc3)  # type: ignore
hedge_in_hits = sum(
    1 for s in kw_sentences
    if any(h in s["sentence"].lower() for h in ["would","could","might","may","perhaps","possibly"])
)
metric_card(f"{round(float(hedge_in_hits)/max(len(kw_sentences),1)*100,0):.0f}%",  # type: ignore
            "Sentences with modal hedging", oc4)

st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_labels = ["🧬 X-Ray", "🕸 Spiderweb", "🫧 Topic Decoder",
              "📈 Timeline", "⚡ Threat Assessor", "📋 Raw Corpus"]
tabs = st.tabs(tab_labels)


# ════════════════════════════════════════════════════════════
# MODULE 1 — SYNTACTIC X-RAY
# ════════════════════════════════════════════════════════════
with tabs[0]:
    section_header("🧬", "Syntactic X-Ray", "Dependency Parsing · spaCy")
    st.markdown(f"""<p style="color:#718096;font-size:0.83rem;margin-bottom:18px">
    Dependency-parse sentences containing <strong style="color:#fbd38d">"{keyword}"</strong> to reveal whether it’s surrounded by
    logistical verbs, modal hedges, or subordinate clauses — the hallmarks of a pragmatic cipher.
    </p>""", unsafe_allow_html=True)

    if not mod_xray:
        st.info("Module disabled in sidebar.")
    else:
        try:
            xr = SyntacticXRay()
            prog_xray = st.progress(0, text="Initialising spaCy parser…")
            prog_xray.progress(20, text="Loading language model…")
            xdata = xr.analyse(keyword, corpus=all_emails)
            prog_xray.progress(100, text="Parsing complete.")
            time.sleep(0.3)
            prog_xray.empty()

            x1, x2, x3, x4 = st.columns(4)
            metric_card(xdata["total_emails_with_keyword"], "Emails hit", x1)
            metric_card(xdata["sentences_analysed"], "Sentences parsed", x2)
            metric_card(f'{xdata["hedge_percentage"]}%', "Modal hedging", x3)
            metric_card(f'{xdata["logistic_verb_percentage"]}%', "Logistic verbs", x4)

            st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("<p style='color:#a0aec0;font-weight:600;font-size:0.85rem;margin-bottom:8px'>🔴 Top Root Verbs</p>", unsafe_allow_html=True)
                for verb, count in xdata["top_root_verbs"]:
                    pct = round(count / max(xdata["sentences_analysed"], 1) * 100, 1)
                    is_logistic = verb.lower() in ["schedule","transport","arrange",
                                                    "coordinate","deliver","send","move",
                                                    "transfer","book","confirm","organize"]
                    col = "#fc8181" if is_logistic else "#63b3ed"
                    st.markdown(score_bar(f"{'🚨' if is_logistic else '▸'} {verb}", pct, col), unsafe_allow_html=True)

            with col_r:
                st.markdown("<p style='color:#a0aec0;font-weight:600;font-size:0.85rem;margin-bottom:8px'>🔵 Top Modifiers on Keyword</p>", unsafe_allow_html=True)
                for mod, count in xdata["top_modifiers"][:8]:
                    pct = round(count / max(xdata["sentences_analysed"], 1) * 100, 1)
                    st.markdown(score_bar(f"▸ {mod}", pct, "#d6bcfa"), unsafe_allow_html=True)

            st.markdown("<div style='margin:18px 0 10px 0'></div>", unsafe_allow_html=True)
            section_header("📜", "Sentence Dependency Viewer", "Interactive")

            if xdata["sentence_details"]:
                sent_opts = [
                    f'[{i+1}] {d["sender"][:25]} — {d["subject"][:40]}'
                    for i, d in enumerate(xdata["sentence_details"])
                ]
                chosen_idx = st.selectbox("Select sentence to inspect:", range(len(sent_opts)),
                                          format_func=lambda i: sent_opts[i])
                detail = xdata["sentence_details"][chosen_idx]

                st.markdown(f"""
                <div class="email-card">
                    <div class="meta">📧 {detail['email_id']} &nbsp;|&nbsp;
                        <span class="sender">{detail['sender']}</span> &nbsp;|&nbsp;
                        <span class="subject">{detail['subject'][:60]}</span>
                    </div>
                    <div class="body-snippet" style="margin-top:8px">{
                        highlight_keyword(detail["sentence"], keyword)
                    }</div>
                </div>""", unsafe_allow_html=True)

                # Full email overlay
                full_email = cp.get_email_by_id(detail["email_id"])
                if full_email:
                    render_email_overlay(full_email, keyword, key_suffix=f"xray_{chosen_idx}")

                st.markdown("<p style='color:#718096;font-size:0.78rem;margin:10px 0 5px'>Token / POS tags:</p>", unsafe_allow_html=True)
                pills_html = "".join(pos_pill(t["text"], t["pos"], t["is_keyword"]) for t in detail["tokens"])
                st.markdown(f'<div style="line-height:2.2">{pills_html}</div>', unsafe_allow_html=True)

                if detail["edges"]:
                    import pandas as pd  # type: ignore
                    edges_df = pd.DataFrame(detail["edges"])
                    st.markdown("<p style='color:#718096;font-size:0.78rem;margin:14px 0 5px'>Dependency edges:</p>", unsafe_allow_html=True)
                    st.dataframe(
                        edges_df.rename(columns={"from":"Head","to":"Dependent","dep":"Relation"}),
                        use_container_width=True, hide_index=True,
                    )
            else:
                st.warning(f"No sentences parsed for '{keyword}'. Try a different keyword.")

        except RuntimeError as e:
            st.warning(f"⚠️ {e}")
            st.markdown("**Fallback — Keyword Concordance (no spaCy required):**")
            for row in kw_sentences[:15]:
                st.markdown(f"""
                <div class="email-card">
                    <div class="meta"><span class="sender">{row['sender']}</span></div>
                    <div class="body-snippet">{highlight_keyword(row['sentence'], keyword)}</div>
                </div>""", unsafe_allow_html=True)
                full_email = cp.get_email_by_id(row["id"])
                if full_email:
                    render_email_overlay(full_email, keyword)

# ════════════════════════════════════════════════════════════
# MODULE 2 — ENTITY SPIDERWEB
# ════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("🕸", "Entity Spiderweb", "NER · NetworkX · Co-occurrence")
    st.markdown(f"""<p style="color:#718096;font-size:0.83rem;margin-bottom:18px">
    NLTK Named Entity Recognition extracts <b>PERSON</b>, <b>ORGANIZATION</b>, and <b>LOCATION</b>
    tags from every email containing <strong style="color:#fbd38d">"{keyword}"</strong>. Edge weights
    represent co-occurrence frequency — mathematically proving the logistical supply chain.
    <br><span style="color:#d69e2e;font-size:0.78rem">💡 Click a node in the graph to see emails linked to that entity below.</span>
    </p>""", unsafe_allow_html=True)

    if not mod_spider:
        st.info("Module disabled in sidebar.")
    else:
        spider_col1, spider_col2 = st.columns([3, 1])
        with spider_col1:
            spider_emails = st.slider("Max emails to process (NER is slow):", 20, 300, 80, key="spider_n")
        with spider_col2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            run_spider = st.button("⚙️ Build Entity Graph", key="btn_spider")

        if run_spider or st.session_state.get("spider_done_kw") == keyword:
            st.session_state["spider_done_kw"] = keyword
            try:
                es   = EntitySpiderweb()

                # ── Progress bar for NER (chunked loop simulation) ──────────
                prog_spider = st.progress(0, text="Initialising NLTK NER engine…")
                prog_spider.progress(10, text="Loading corpus…")
                time.sleep(0.1)

                # Run the analysis (the heavy part)
                prog_spider.progress(20, text=f"Running NER on up to {spider_emails} emails…")
                sdata = es.analyse(keyword, max_emails=spider_emails, corpus=all_emails)
                prog_spider.progress(80, text="Building co-occurrence graph…")
                time.sleep(0.15)
                prog_spider.progress(100, text="Graph complete.")
                time.sleep(0.3)
                prog_spider.empty()

                s1, s2, s3 = st.columns(3)
                metric_card(sdata["total_emails"],     "Emails processed",      s1)
                metric_card(len(sdata["nodes"]),       "Named entities found",  s2)
                metric_card(len(sdata["edges"]),       "Co-occurrence edges",   s3)

                if not sdata["nodes"]:
                    st.warning("No named entities extracted. Try a broader keyword.")
                else:
                    col_l, col_r = st.columns([2, 1])

                    with col_r:
                        st.markdown("<p style='color:#a0aec0;font-weight:600;font-size:0.85rem;margin-bottom:8px'>🏆 Top Connected Entities</p>", unsafe_allow_html=True)
                        max_weight = sdata["top_entities"][0][1] if sdata["top_entities"] else 1
                        for name, weight in sdata["top_entities"]:
                            pct = round(weight / max(max_weight, 1) * 100, 1)
                            st.markdown(score_bar(name[:30], pct, "#68d391"), unsafe_allow_html=True)

                        # Entity selector for overlay
                        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
                        st.markdown("<p style='color:#a0aec0;font-weight:600;font-size:0.85rem;margin-bottom:6px'>🔍 Click node to explore emails</p>", unsafe_allow_html=True)
                        entity_names = [n["id"] for n in sdata["nodes"][:50]]
                        selected_entity = st.selectbox(
                            "Select entity:", ["— pick an entity —"] + entity_names,
                            key="spider_entity_sel"
                        )

                    with col_l:
                        try:
                            from streamlit_agraph import agraph, Node, Edge, Config  # type: ignore
                            type_colors = {
                                "PERSON":       "#63b3ed",
                                "ORGANIZATION": "#fc8181",
                                "GPE":          "#68d391",
                                "LOCATION":     "#d6bcfa",
                                "FACILITY":     "#fbd38d",
                                "":             "#a0aec0"
                            }
                            nodes = [Node(
                                id=n["id"], label=n["id"][:18],
                                size=20, color=type_colors.get(n["type"], "#a0aec0"),
                                font={"color": "#1e293b", "size": 11, "face": "Inter"}
                            ) for n in sdata["nodes"][:60]]
                            edges = [Edge(
                                source=e["source"], target=e["target"],
                                width=min(e["weight"] * 0.8 + 1, 6),
                                color="rgba(99,179,237,0.35)"
                            ) for e in sdata["edges"][:80]]
                            config = Config(
                                width="100%", height=500,
                                directed=False, physics=True,
                                backgroundColor="#ffffff",
                                nodeHighlightBehavior=True,
                                highlightColor="#3182ce",
                            )
                            # The agraph returns user selection but since streamlit_agraph rerender might be slow
                            agraph(nodes=nodes, edges=edges, config=config)

                            # Reset view button
                            if st.button("🔄 Reset Graph View", key="spider_reset"):
                                st.session_state["spider_done_kw"] = keyword
                                st.rerun()

                        except ImportError:
                            st.info("💡 Install `streamlit-agraph` for interactive graph: `pip install streamlit-agraph`")
                            import pandas as pd  # type: ignore
                            df_edges = pd.DataFrame(sdata["edges"][:30]).sort_values("weight", ascending=False)
                            df_edges.columns = ["Entity A", "Entity B", "Co-occurrence Weight"]
                            st.dataframe(df_edges, use_container_width=True, hide_index=True)

                        legend_html = """
                        <div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;font-size:0.75rem">
                            <span>🔵 <span style="color:#63b3ed">PERSON</span></span>
                            <span>🔴 <span style="color:#fc8181">ORGANIZATION</span></span>
                            <span>🟢 <span style="color:#68d391">GPE/Location</span></span>
                            <span>🟣 <span style="color:#d6bcfa">FACILITY</span></span>
                        </div>"""
                        st.markdown(legend_html, unsafe_allow_html=True)

                    # ── Node click → email overlay ────────────────────────────
                    if selected_entity and selected_entity != "— pick an entity —":
                        node_emails = sdata.get("node_email_map", {}).get(selected_entity, [])
                        st.markdown("---")
                        st.markdown(f"""
                        <div class="node-hint">
                            🕸 Showing <strong>{len(node_emails)}</strong> emails mentioning entity:
                            <strong style="color:#3182ce">{selected_entity}</strong>
                        </div>""", unsafe_allow_html=True)

                        if node_emails:
                            for nem in node_emails[:10]:
                                body_snip = highlight_keyword(nem["body"], keyword, 300)
                                st.markdown(f"""
                                <div class="email-card">
                                    <div class="meta">
                                        <code style="color:#81e6d9;font-size:0.72rem">{nem['id']}</code> &nbsp;|&nbsp;
                                        <span class="sender">{nem.get('sender','Unknown')}</span> &nbsp;|&nbsp;
                                        <span class="subject">{nem.get('subject','')[:55]}</span>
                                    </div>
                                    <div class="body-snippet" style="margin-top:8px">{body_snip}</div>
                                </div>""", unsafe_allow_html=True)
                                render_email_overlay(nem, keyword, key_suffix=f"spider_{nem['id']}")
                        else:
                            st.markdown('<div class="email-card" style="color:#718096;text-align:center;padding:20px">No email data cached for this entity.</div>', unsafe_allow_html=True)

            except RuntimeError as e:
                st.warning(f"⚠️ {e}")
        else:
            st.markdown("""
            <div class="email-card" style="text-align:center;padding:32px;color:#718096">
                Click <strong style="color:#63b3ed">Build Entity Graph</strong> to run NER analysis.<br>
                <span style="font-size:0.78rem">Processing time depends on number of emails selected.</span>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MODULE 3 — CONTEXT DECODER (LDA)
# ════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("🫧", "Hidden Context Decoder", "Unsupervised LDA · Gensim")
    st.markdown(f"""<p style="color:#718096;font-size:0.83rem;margin-bottom:18px">
    Latent Dirichlet Allocation clusters background vocabulary in emails containing
    <strong style="color:#fbd38d">"{keyword}"</strong> into hidden semantic groups —
    statistically proving whether the network discusses culinary themes or logistical coordination.
    {"<br><span style='color:#63b3ed;font-size:0.78rem'>🤖 AI topic interpretation enabled — each topic will receive a forensic description.</span>" if use_ai_topics else ""}
    </p>""", unsafe_allow_html=True)

    if not mod_topic:
        st.info("Module disabled in sidebar.")
    else:
        run_lda = st.button("⚙️ Run LDA Topic Modeling", key="btn_lda")

        if run_lda or st.session_state.get("lda_done_kw") == keyword:
            st.session_state["lda_done_kw"] = keyword
            try:
                cd = ContextDecoder()

                prog_lda = st.progress(0, text="Preparing corpus for LDA…")
                prog_lda.progress(15, text="Tokenising and filtering stopwords…")

                # Step-through progress for LDA
                prog_lda.progress(30, text=f"Building dictionary with {num_topics} topics…")
                time.sleep(0.1)
                prog_lda.progress(50, text="Running LDA passes (this may take ~15s)…")

                tdata = cd.analyse(keyword, num_topics=num_topics,
                                   interpret_topics=use_ai_topics, corpus=all_emails)

                if use_ai_topics:
                    prog_lda.progress(75, text="🤖 AI interpreting topics…")
                    time.sleep(0.2)

                prog_lda.progress(100, text="LDA complete.")
                time.sleep(0.3)
                prog_lda.empty()

                if "error" in tdata:
                    st.error(tdata["error"])
                else:
                    t1, t2 = st.columns(2)
                    metric_card(tdata["emails_analysed"], "Emails in LDA corpus", t1)
                    metric_card(tdata["num_topics"],      "Hidden topics extracted", t2)

                    st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)

                    cols = st.columns(min(num_topics, 3))
                    for i, topic in enumerate(tdata["topics"]):
                        c = cols[i % 3]
                        color = TOPIC_COLOURS[i % len(TOPIC_COLOURS)]
                        words_html = "".join(
                            f'<span style="display:inline-block;margin:3px 3px;padding:4px 10px;border-radius:8px;'
                            f'font-size:0.75rem;background:rgba(255,255,255,0.06);color:{color};'
                            f'border:1px solid rgba(255,255,255,0.1);opacity:{0.5 + (w * 4):.2f}">{word}</span>'
                            for word, w in zip(topic["words"], topic["weights"])
                        )
                        interp = topic.get("interpretation")

                        # Build AI interpretation HTML
                        ai_html = ""
                        if interp:
                            ai_badge = (
                                '<span class="ai-badge">🤖 AI</span>'
                                if interp.get("ai_powered")
                                else '<span class="ai-badge ai-badge-fallback">RULE</span>'
                            )
                            ai_html = f"""
<div class="ai-interp-card">
    {ai_badge}
    <span style="font-weight:700;color:#2d3748;margin-left:6px">{interp.get('best_label','')}</span><br>
    <span style="color:#4a5568;font-size:0.8rem">{interp.get('description','')}</span><br>
    <span style="font-size:0.78rem;margin-top:4px;display:block">{interp.get('forensic_significance','')}</span>
</div>"""

                        c.markdown(f"""
<div class="email-card" style="border-color:{color}40;text-align:center">
    <div style="color:{color};font-weight:700;font-size:0.9rem;margin-bottom:8px">
        ◉ {topic["label"]}
    </div>
    <div style="line-height:2.2">{words_html}</div>
    {ai_html}
</div>""", unsafe_allow_html=True)

                    # Topic distribution chart
                    if tdata["doc_topics"]:
                        import pandas as pd  # type: ignore
                        from collections import Counter as Ctr
                        dom_counts = Ctr(d["dominant_topic"] for d in tdata["doc_topics"])
                        df_dist = pd.DataFrame([
                            {"Topic": f'Topic {int(k)+1}', "Emails": v}  # type: ignore
                            for k, v in sorted(dom_counts.items())
                        ])
                        st.markdown("<div style='margin:18px 0 8px 0'></div>", unsafe_allow_html=True)
                        section_header("📊", "Topic Document Distribution", "")
                        st.bar_chart(df_dist.set_index("Topic"), use_container_width=True, color="#63b3ed")

            except RuntimeError as e:
                st.warning(f"⚠️ {e}")
                st.markdown("**Fallback — Top contextual words (no Gensim):**")
                from collections import Counter
                import re as _re
                stop = ContextDecoder._STOP
                all_words = []
                for e in kw_hits[:100]:
                    tokens = _re.findall(r'\b[a-z]{4,}\b', e["body"].lower())
                    all_words.extend(t for t in tokens if t not in stop and t != keyword.lower())
                top_words = Counter(all_words).most_common(20)
                pills = "".join(
                    f'<span class="wl-tag">{w} <span style="color:#718096">×{c}</span></span>'
                    for w, c in top_words
                )
                st.markdown(f'<div style="margin-top:8px">{pills}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="email-card" style="text-align:center;padding:32px;color:#718096">
                Click <strong style="color:#63b3ed">Run LDA Topic Modeling</strong> to decode hidden contexts.<br>
                <span style="font-size:0.78rem">Requires Gensim. Adjust topic count in the sidebar.</span>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MODULE 4 — DIACHRONIC TIMELINE
# ════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("📈", "Diachronic Timeline", "Syntactic Shift · Pandas")
    st.markdown(f"""<p style="color:#718096;font-size:0.83rem;margin-bottom:18px">
    Track how <strong>modal hedging density</strong> and <strong>logistical verb density</strong>
    evolve across emails containing <strong style="color:#fbd38d">"{keyword}"</strong>.
    Spikes in hedging correlate with periods of external pressure or legal scrutiny.
    </p>""", unsafe_allow_html=True)

    if not mod_timeline:
        st.info("Module disabled in sidebar.")
    else:
        try:
            dt = DiachronicTimeline()
            prog_tl = st.progress(0, text="Computing time-series metrics…")
            prog_tl.progress(40, text="Calculating hedge and logistic densities…")
            tlinedata = dt.analyse(keyword, annotated_corpus=annotated_emails)
            prog_tl.progress(80, text="Computing rolling averages…")
            time.sleep(0.1)
            prog_tl.progress(100, text="Timeline ready.")
            time.sleep(0.2)
            prog_tl.empty()

            if "error" in tlinedata:
                st.warning(tlinedata["error"])
            else:
                import pandas as pd  # type: ignore

                t1, t2, t3, t4 = st.columns(4)
                metric_card(tlinedata["total_emails"],                    "Emails in timeline",    t1)
                metric_card(f'{tlinedata["avg_hedge_density"]}%',         "Avg hedge density",     t2)
                metric_card(f'{tlinedata["avg_logistic_density"]}%',      "Avg logistic density",  t3)
                peak = tlinedata["peak_hedge_email"]
                metric_card(f'{peak["hedge_density"]}%', "Peak hedge density", t4)

                df = pd.DataFrame(tlinedata["timeline"])

                st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)
                section_header("📊", "Hedge & Logistic Density Over Email Sequence", "")

                chart_col, info_col = st.columns([3, 1])
                with chart_col:
                    st.line_chart(
                        df.set_index("index")[["hedge_rolling", "logistic_rolling"]],
                        use_container_width=True,
                        color=["#fc8181", "#63b3ed"],
                    )

                with info_col:
                    st.markdown("<p style='color:#a0aec0;font-size:0.8rem;font-weight:600'>Legend</p>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="font-size:0.78rem;line-height:2">
                        <span style="color:#fc8181">—</span> Modal Hedging (10-email rolling avg)<br>
                        <span style="color:#63b3ed">—</span> Logistic Verbs (10-email rolling avg)
                    </div>""", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="margin-top:14px">
                        <p style="color:#a0aec0;font-size:0.78rem;font-weight:600">Peak Hedging Email</p>
                        <div style="font-size:0.75rem;color:#718096;line-height:1.7">
                            ID: <span style="color:#fbd38d">{peak.get("email_id","?")[:15]}</span><br>
                            Sender: {peak.get("sender","?")[:20]}<br>
                            Hedge: <span style="color:#fc8181">{peak["hedge_density"]}%</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Flag count scatter
                st.markdown("<div style='margin:18px 0 8px 0'></div>", unsafe_allow_html=True)
                section_header("📌", "Flagged Term Density", "All terms across email sequence")
                st.bar_chart(df.set_index("index")["flag_count"],
                             use_container_width=True, color="#d6bcfa")

                # ── Raw timeline table with email overlay ──────────────────
                with st.expander("📄 View raw timeline data"):
                    show_cols = ["email_id","sender","subject",
                                 "hedge_density","logistic_density",
                                 "hedge_rolling","logistic_rolling","flag_count"]
                    st.dataframe(
                        df[show_cols].rename(columns={
                            "email_id":"Email ID","hedge_density":"Hedge %",
                            "logistic_density":"Logistic %","hedge_rolling":"Hedge Rolling",
                            "logistic_rolling":"Logistic Rolling","flag_count":"Flags"
                        }),
                        use_container_width=True, hide_index=True
                    )

                st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
                st.markdown("<p style='color:#a0aec0;font-size:0.8rem;font-weight:600'>📧 Open email by ID</p>", unsafe_allow_html=True)
                timeline_ids = df["email_id"].tolist()
                selected_tl_id = st.selectbox(
                    "Select email ID:", ["— pick —"] + timeline_ids,
                    key="tl_email_sel"
                )
                if selected_tl_id and selected_tl_id != "— pick —":
                    tl_email = cp.get_email_by_id(selected_tl_id)
                    if tl_email:
                        st.markdown(f"""
                        <div class="email-card" style="margin-top:10px">
                            <div class="meta">
                                <code style="color:#81e6d9;font-size:0.72rem">{tl_email['id']}</code> &nbsp;|&nbsp;
                                <span class="sender">{tl_email.get('sender','')}</span> &nbsp;|&nbsp;
                                <span class="subject">{tl_email.get('subject','')[:60]}</span>
                            </div>
                            <div class="body-snippet" style="margin-top:8px">
                                {highlight_keyword(tl_email.get('body',''), keyword, 300)}
                            </div>
                        </div>""", unsafe_allow_html=True)
                        render_email_overlay(tl_email, keyword, key_suffix=f"tl_{selected_tl_id}")
                    else:
                        st.warning("Email not found in corpus.")

        except RuntimeError as e:
            st.warning(f"⚠️ {e}")

# ════════════════════════════════════════════════════════════
# MODULE 5 — THREAT ASSESSOR
# ════════════════════════════════════════════════════════════
with tabs[4]:
    section_header("⚡", "Threat Assessor", "Zero-Shot Classification · HuggingFace")
    st.markdown("""<p style="color:#718096;font-size:0.83rem;margin-bottom:18px">
    A HuggingFace Transformer classifies each email’s pragmatic intent into forensic labels
    with a confidence percentage — <em>objectively</em>, without human bias or assumption.
    </p>""", unsafe_allow_html=True)

    if not mod_threat:
        st.info("Module disabled in sidebar.")
    else:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown("<p style='color:#a0aec0;font-size:0.8rem;margin-bottom:6px'>Custom forensic labels (one per line):</p>", unsafe_allow_html=True)
            labels_input = st.text_area(
                "Labels", value="\n".join(FORENSIC_LABELS),
                height=160, label_visibility="collapsed", key="labels_ta"
            )
        with col_b:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            single_mode = st.toggle("Single-email mode", value=False)
            n_assess    = st.slider("Emails to assess (batch):", 3, 50, 8, key="n_assess")

        custom_labels = [l.strip() for l in labels_input.strip().split("\n") if l.strip()]

        if single_mode:
            if kw_hits:
                email_opts = [
                    f'{e["id"][:15]} — {e["sender"][:25]} — {e["subject"][:35]}'
                    for e in kw_hits[:50]
                ]
                chosen = st.selectbox("Choose email:", range(len(email_opts)),
                                      format_func=lambda i: email_opts[i])
                target_email = kw_hits[chosen]
                full_body_display = highlight_keyword(target_email['body'], keyword, len(target_email['body']))
                
                st.markdown(f"""
                <div class="email-card">
                    <div class="meta"><span class="sender">{target_email['sender']}</span> · {target_email['subject']}</div>
                    <div class="body-snippet">{highlight_keyword(target_email['body'], keyword, 500)}</div>
                    <details style="margin-top:12px; border-top:1px solid var(--border); padding-top:10px;">
                        <summary style="cursor:pointer; color:var(--accent); font-weight:600; font-size:0.85rem; outline:none;">
                            View Full Email
                        </summary>
                        <div class="overlay-body" style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:16px; font-size:0.88rem; line-height:1.7; max-height:350px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; margin-top:8px;">{full_body_display}</div>
                    </details>
                </div>""", unsafe_allow_html=True)

                if st.button("⚡ Assess This Email", key="btn_assess_single"):
                    try:
                        ta = ThreatAssessor()
                        prog_ta = st.progress(0, text="Loading transformer model (first run ~30s)…")
                        prog_ta.progress(30, text="Model loaded. Running classification…")
                        result  = ta.assess_email(target_email["body"], custom_labels)
                        prog_ta.progress(100, text="Classification complete.")
                        time.sleep(0.2)
                        prog_ta.empty()

                        label_colors = {
                            "Logistical Coordination":  "#63b3ed",
                            "Veiled Coercion":          "#fc8181",
                            "Social Grooming":          "#fbd38d",
                            "Financial Transaction":    "#68d391",
                            "Routine Communication":    "#a0aec0",
                            "Information Suppression":  "#d6bcfa",
                            "Coded Language Usage":     "#f6ad55",
                        }
                        
                        badge = threat_badge(result['top_label'], result['top_score'])
                        score_bars_html = "".join(
                            score_bar(s["label"], s["score"], label_colors.get(s["label"], "#63b3ed"))
                            for s in result["all_scores"]
                        )
                        st.markdown(f"""
                        <div class="email-card" style="border-color:var(--accent); margin-top:-8px;">
                            <div style="margin-bottom:12px">{badge}</div>
                            <strong style="font-size:0.85rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em;">All Threat Scores</strong>
                            <div style="margin-top:8px;">{score_bars_html}</div>
                        </div>""", unsafe_allow_html=True)
                    except RuntimeError as e:
                        st.warning(f"⚠️ {e}")
            else:
                st.warning(f"No emails found for '{keyword}'.")

        else:
            # ── Batch mode ────────────────────────────────────────────────────
            if st.button(f"⚡ Assess Top {n_assess} Emails", key="btn_assess_batch"):
                try:
                    ta      = ThreatAssessor()
                    results = []
                    prog_batch = st.progress(0, text="Loading transformer model…")
                    pipe = ta.get_pipeline()
                    prog_batch.progress(10, text="Model ready. Starting batch classification…")

                    batch_target = kw_hits[:n_assess]
                    for i, e in enumerate(batch_target):
                        pct = int(10 + (i + 1) / len(batch_target) * 85)
                        prog_batch.progress(pct, text=f"Assessing email {i+1}/{len(batch_target)}: {e.get('sender','')[:30]}…")
                        res = ta.assess_email(e["body"], custom_labels)
                        results.append({**e, **res})

                    prog_batch.progress(100, text="Batch complete.")
                    time.sleep(0.3)
                    prog_batch.empty()

                    st.session_state["threat_results"] = results

                except RuntimeError as e:
                    st.warning(f"⚠️ {e}")
                    st.markdown("**Fallback — Rule-based intent signals:**")
                    results = []
                    for e in kw_hits[:n_assess]:
                        body_lower = e["body"].lower()
                        signals = []
                        if any(v in body_lower for v in ["schedule","arrange","transport","book"]):
                            signals.append("🚨 Logistical Coordination")
                        if any(h in body_lower for h in ["would","could","if you","perhaps"]):
                            signals.append("🔶 Modal Hedging Detected")
                        if any(f in body_lower for f in active_food):
                            signals.append("🟡 Food Code Word Present")
                        if any(eu in body_lower for eu in active_euph):
                            signals.append("🔴 Euphemism Present")
                        signals_html = " ".join(
                            f'<span class="wl-tag">{s}</span>' for s in signals
                        ) or "<span style='color:#718096'>No strong signals</span>"
                        results.append({**e, "signals_html": signals_html, "fallback": True})  # type: ignore
                    st.session_state["threat_results"] = results

            # ── Display stored results ────────────────────────────────────────
            if "threat_results" in st.session_state:
                for idx, res in enumerate(st.session_state["threat_results"]):
                    label_colors = {
                        "Logistical Coordination":  "#63b3ed",
                        "Veiled Coercion":          "#fc8181",
                        "Social Grooming":          "#fbd38d",
                        "Financial Transaction":    "#68d391",
                        "Routine Communication":    "#a0aec0",
                        "Information Suppression":  "#d6bcfa",
                        "Coded Language Usage":     "#f6ad55",
                    }

                    full_email_obj = cp.get_email_by_id(res.get("id", "")) or res
                    email_body_display = highlight_keyword(full_email_obj.get("body", ""), keyword, len(full_email_obj.get("body", "")))

                    if res.get("fallback"):
                        st.markdown(f"""
                        <div class="email-card">
                            <div class="meta">
                                <code style="color:#81e6d9;font-size:0.72rem">{res.get('id', 'N/A')}</code> &nbsp;|&nbsp;
                                <span class="sender">{res['sender']}</span> · {res['subject'][:50]}
                            </div>
                            <div style="margin:6px 0">{res.get('signals_html','')}</div>
                            <div class="body-snippet">{highlight_keyword(res['body'], keyword, 200)}</div>
                            <details style="margin-top:12px; border-top:1px solid var(--border); padding-top:10px;">
                                <summary style="cursor:pointer; color:var(--accent); font-weight:600; font-size:0.85rem; outline:none;">
                                    View Full Email
                                </summary>
                                <div class="overlay-body" style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:16px; font-size:0.88rem; line-height:1.7; max-height:350px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; margin-top:10px;">{email_body_display}</div>
                            </details>
                        </div>""", unsafe_allow_html=True)
                    else:
                        badge = threat_badge(res["top_label"], res["top_score"])
                        score_bars_html = "".join(
                            score_bar(s["label"], s["score"],
                                      label_colors.get(s["label"], "#63b3ed"))
                            for s in res["all_scores"]
                        )
                        st.markdown(f"""
                        <div class="email-card">
                            <div class="meta">
                                <code style="color:#81e6d9;font-size:0.72rem">{res['id']}</code> &nbsp;|&nbsp;
                                <span class="sender">{res['sender']}</span> ·
                                <span class="subject">{res['subject'][:60]}</span>
                            </div>
                            <div style="margin:6px 0">{badge}</div>
                            <div class="body-snippet">{highlight_keyword(res['body'], keyword, 250)}</div>
                            <details style="margin-top:12px; border-top:1px solid var(--border); padding-top:10px;">
                                <summary style="cursor:pointer; color:var(--accent); font-weight:600; font-size:0.85rem; outline:none;">
                                    View Full Details & Scores
                                </summary>
                                <div style="margin-top: 12px; margin-bottom: 16px;">
                                    <strong style="font-size:0.85rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em;">All Threat Scores</strong>
                                    <div style="margin-top: 8px;">{score_bars_html}</div>
                                </div>
                                <strong style="font-size:0.85rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em;">Full Email Body</strong>
                                <div class="overlay-body" style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:16px; font-size:0.88rem; line-height:1.7; max-height:350px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; margin-top:8px;">{email_body_display}</div>
                            </details>
                        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MODULE 6 — RAW CORPUS BROWSER
# ════════════════════════════════════════════════════════════
with tabs[5]:
    section_header("📋", "Raw Corpus Browser", f'Filtered to: "{keyword}"')
    st.markdown(f"""<p style="color:#718096;font-size:0.83rem;margin-bottom:18px">
    Browse all <strong style="color:#63b3ed">{len(kw_hits)}</strong> emails containing
    <strong style="color:#fbd38d">"{keyword}"</strong>. Use the search to refine further.
    </p>""", unsafe_allow_html=True)

    sub_search = st.text_input(
        "🔎 Filter further by secondary term:",
        placeholder="e.g. 'schedule', 'flight'…", key="sub_search"
    )
    show_emails = kw_hits
    if sub_search.strip():
        sub = sub_search.strip().lower()
        show_emails = [
            e for e in kw_hits
            if sub in e.get("body","").lower() or sub in e.get("subject","").lower()
        ]
        st.markdown(f"<p style='color:#718096;font-size:0.78rem'>Showing {len(show_emails)} matches for '{sub_search}'</p>", unsafe_allow_html=True)

    page_size   = 20
    total_pages = max(1, (len(show_emails) + page_size - 1) // page_size)
    page        = st.slider("Page:", 1, total_pages, 1, key="corpus_pg") if total_pages > 1 else 1
    page_emails = show_emails[(page-1)*page_size : page*page_size]  # type: ignore

    for email in page_emails:
        body_snip = highlight_keyword(email["body"], keyword, 350)
        st.markdown(f"""
        <div class="email-card">
            <div class="meta">
                📧 <code style="color:#81e6d9;font-size:0.72rem">{email['id']}</code> &nbsp;|&nbsp;
                <span class="sender">{email.get('sender','Unknown')}</span> &nbsp;|&nbsp;
                <span class="subject">{email.get('subject','(no subject)')[:70]}</span>
            </div>
            <div class="body-snippet" style="margin-top:8px">{body_snip}</div>
        </div>""", unsafe_allow_html=True)
        render_email_overlay(email, keyword, key_suffix=f"raw_{email['id']}")

    if not page_emails:
        st.markdown('<div class="email-card" style="text-align:center;padding:28px;color:#718096">No emails match this filter combination.</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px;padding:22px;border-top:1px solid rgba(99,179,237,0.12);text-align:center">
    <span style="color:#4a5568;font-size:0.75rem">
        🔬 Forensic Entity-Action Profiler &nbsp;·&nbsp;
        Ameen K.P | EFLU · Data: jmail.world FOIA corpus
    </span>
</div>
""", unsafe_allow_html=True)
