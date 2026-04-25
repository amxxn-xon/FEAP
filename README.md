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

*This command installs the stable PyTorch CPU engine, Transformers, spaCy, and all analytical dependencies.*

### 3. Model Setup
Download the core English language model for the Syntactic X-Ray:

```powershell
python -m spacy download en_core_web_sm
```

### 4. Run the Dashboard
Launch the interface with:

```powershell
streamlit run app.py
```

---

## 🧩 Forensic Architecture

The application is structured into five core analytical modules:

1.  **🧬 Syntactic X-Ray**: Dependency-parsing to uncover the grammatical structure of coded communications.
2.  **🕸 Entity Spiderweb**: Interactive co-occurrence mapping of Persons, Organisations, and Locations.
3.  **🫧 Topic Decoder**: Unsupervised LDA modeling to find hidden thematic contexts.
4.  **📈 Diachronic Timeline**: Time-series analysis of linguistic density and modal hedging.
5.  **⚡ Threat Assessor**: Deep-learning intent classification (Zero-shot) to score illicit patterns.

---

## 📁 Project Structure
- `app.py`: The Streamlit dashboard (UI).
- `engine.py`: The backend processing engine (OOP).
- `data/`: Contains the cleaned and annotated corpus JSON files.
- `.streamlit/config.toml`: Custom configuration to resolve PyTorch/Streamlit watcher conflicts.

---
**Author**: Ameen K.P | EFLU  Hyderabad

