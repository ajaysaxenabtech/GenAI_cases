# Project Structure for Local GenAI-based Industry Fundamental Analysis

# ───────────────────────────────────────────────────────────────────────────────
# Directory Structure
# ───────────────────────────────────────────────────────────────────────────────
# genai_fundamental_analysis/
# ├── data_sources/
# │   ├── pdfs/
# │   ├── excels/
# │   └── api/
# ├── configs/
# │   └── config.yaml
# ├── embeddings/
# ├── models/
# ├── scripts/
# │   ├── ingest.py
# │   ├── embed.py
# │   ├── query.py
# │   └── interface.py
# ├── main.py
# └── README.md

# ───────────────────────────────────────────────────────────────────────────────
# README.md
# ───────────────────────────────────────────────────────────────────────────────

# 🔍 GenAI Fundamental Industry Analysis (Local RAG Pipeline)

A lightweight, locally deployable GenAI pipeline that performs **fundamental analysis** of industries by analyzing **PDFs, Excel sheets, and API sources** assigned by the user. This project leverages Retrieval-Augmented Generation (RAG) with local LLMs and embedding models.

---

## 🧠 Features
- Ingest data from **PDF**, **Excel**, and **API** sources
- Preprocess and store data as text
- Generate vector embeddings locally with `sentence-transformers`
- Use lightweight **LLMs** such as `phi-3-mini` or `mistral-7b`
- Contextual query answering with **RAG (LangChain / LlamaIndex)**
- Simple **CLI or Streamlit interface** for industry-specific analysis

---

## 🧱 Project Structure
```
genai_fundamental_analysis/
├── data_sources/        # Raw data (PDF, Excel, API)
├── configs/             # Configuration YAML
├── embeddings/          # Vector store and raw texts
├── models/              # Local models (optional)
├── scripts/             # Modular Python scripts
├── main.py              # Entry point
└── README.md            # This file
```

---

## ⚙️ Setup Instructions
```bash
# 1. Clone the repository
$ git clone https://github.com/your-username/genai_fundamental_analysis.git
$ cd genai_fundamental_analysis

# 2. Install dependencies
$ pip install -r requirements.txt

# 3. Run the pipeline
$ python main.py
```

---

## 🛠️ Tools & Frameworks
- **LLM**: `phi-3-mini` / `mistral-7b` via `llama.cpp`
- **Embeddings**: `BAAI/bge-small-en` or `sentence-transformers`
- **Vector DB**: `FAISS` / `Chroma`
- **Agentic / RAG Framework**: `LangChain` or `LlamaIndex`
- **Interface**: `Streamlit` (optional)

---

## ✅ Use Case Example
> “Analyze all uploaded PDFs and Excel files related to the Pharma sector. Summarize key risks, growth indicators, and recent developments.”

The system will:
1. Extract and clean data from all assigned sources
2. Create vector embeddings
3. Use a local LLM to retrieve relevant chunks and generate structured insights

---

## 💬 Contributing
Feel free to fork, raise issues, or open PRs. For questions or enhancements, reach out via GitHub Discussions or connect with me on [LinkedIn](https://www.linkedin.com/).

---

## 📄 License
MIT License

---

## 🙌 Acknowledgements
- [LangChain](https://www.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [BAAI/bge](https://huggingface.co/BAAI/bge-small-en)

# ───────────────────────────────────────────────────────────────────────────────
# config.yaml (inside configs/)
# ───────────────────────────────────────────────────────────────────────────────

# Example YAML
model:
  embedding_model: "BAAI/bge-small-en"
  llm_model: "phi-3-mini"

paths:
  pdf_dir: "data_sources/pdfs"
  excel_dir: "data_sources/excels"
  vector_store: "embeddings/vector_db"

parameters:
  chunk_size: 500
  chunk_overlap: 50

# ───────────────────────────────────────────────────────────────────────────────
# main.py (entry point)
# ───────────────────────────────────────────────────────────────────────────────

from scripts.ingest import ingest_sources
from scripts.embed import generate_embeddings
from scripts.query import ask_question
from scripts.interface import run_interface

if __name__ == "__main__":
    print("\nStep 1: Ingesting data...")
    ingest_sources()

    print("\nStep 2: Generating embeddings...")
    generate_embeddings()

    print("\nStep 3: Launching interface...")
    run_interface()

# ───────────────────────────────────────────────────────────────────────────────
# scripts/ingest.py
# ───────────────────────────────────────────────────────────────────────────────

import os
import glob
import pandas as pd
import fitz  # PyMuPDF
from pathlib import Path

RAW_TEXT_DIR = "embeddings/raw_texts"
Path(RAW_TEXT_DIR).mkdir(parents=True, exist_ok=True)

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def extract_excel_text(excel_path):
    dfs = pd.read_excel(excel_path, sheet_name=None)
    return "\n".join(df.to_string() for df in dfs.values())

def ingest_sources():
    pdfs = glob.glob("data_sources/pdfs/*.pdf")
    excels = glob.glob("data_sources/excels/*.xlsx")

    for path in pdfs:
        name = os.path.splitext(os.path.basename(path))[0]
        text = extract_pdf_text(path)
        with open(f"{RAW_TEXT_DIR}/{name}.txt", "w", encoding="utf-8") as f:
            f.write(text)

    for path in excels:
        name = os.path.splitext(os.path.basename(path))[0]
        text = extract_excel_text(path)
        with open(f"{RAW_TEXT_DIR}/{name}.txt", "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Ingested {len(pdfs)} PDFs and {len(excels)} Excel files.")
