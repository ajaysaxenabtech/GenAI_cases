
```
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
# └── main.py

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
