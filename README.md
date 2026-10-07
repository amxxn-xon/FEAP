# 🔬 Forensic Entity-Action Profiler (FEAP)

An advanced Object-Oriented NLP dashboard for decoding weaponized language and logistical ciphers within illicit networks, specifically tailored for the Epstein email corpus.

---

## ⚡ Quick Start for Windows

Follow these steps to set up the profiler on any Windows machine.

### 1. Prerequisites
- **Python 3.10 or higher** installed.
- Ensure `pip` is added to your System PATH.

### 2. Installation (The "One Command" Fix)
Open a terminal in the project directory and run:

```powershell
pip install -r requirements.txt
```

*This command installs the stable PyTorch CPU engine, Transformers, spaCy (with its English model), and all analytical dependencies.*

### 3. Model Setup
`requirements.txt` already installs the spaCy English model. If you installed packages another way, download it manually:

```powershell
python -m spacy download en_core_web_sm
```

The zero-shot transformer (`cross-encoder/nli-MiniLM2-L6-H768`) downloads automatically from Hugging Face the first time an AI feature runs.

### 4. Run the Dashboard
Launch the interface with:

```powershell
streamlit run app.py
```

### 5. Run the Tests (optional)

```powershell
python -m pytest
```

---

## 🧩 Forensic Architecture

The application is structured into five core analytical modules plus a corpus browser:

1.  **🧬 Syntactic X-Ray**: spaCy dependency parsing of every sentence containing the keyword — root verbs (incl. phrasal verbs like *pick up*), modifiers attached to the keyword, and modal hedges.
2.  **🕸 Entity Spiderweb**: spaCy NER + NetworkX co-occurrence graph of Persons, Organisations, Locations and Facilities. Click a node to read the emails behind it; *brokers* (betweenness centrality) show who bridges separate clusters.
3.  **🫧 Topic Decoder**: Unsupervised Gensim LDA over the vocabulary *around* the keyword (the keyword itself and e-mail boilerplate are excluded), with optional zero-shot topic labels and the most representative emails per topic.
4.  **📈 Diachronic Timeline**: Hedge / logistic-verb / flagged-term density over time. Dates are recovered from reply and forward headers ("On Apr 6, 2016 … wrote"), so the chart is chronological; corpus order is available as an alternative.
5.  **⚡ Threat Assessor**: Deep-learning intent classification (zero-shot NLI, batched) against editable forensic labels, with a rule-based fallback when the model is unavailable.
6.  **📋 Raw Corpus**: Browse, filter by secondary term or sender, and export matching emails.

### Matching
Keyword search is **whole-word** by default and accepts simple inflections (`model` → *models, modeling*), so `pie` no longer matches *piece* and the hedge `may` no longer matches *mayor* or the month *May*. Switch off **Whole-word match** in the sidebar for plain substring search.

### Exports
Every module offers a **⬇️ Download CSV** button for its results (parsed sentences, entities and edges, topic assignments, timeline, threat scores, filtered emails).

---

## 📁 Project Structure
- `app.py`: The Streamlit dashboard (UI). Analyses are cached per keyword/setting, so interacting with widgets doesn't recompute them.
- `engine.py`: The backend processing engine (OOP). Models are loaded once per process and shared between modules.
- `tests/`: Pytest suite — engine unit tests and a headless dashboard smoke test.
- `data/`: Contains the cleaned and annotated corpus JSON files.
- `.streamlit/config.toml`: Custom configuration to resolve PyTorch/Streamlit watcher conflicts.

---
**Author**: Ameen K.P | EFLU  Hyderabad
