# 🔍 GenAI Fundamental Industry Analysis (Local RAG Pipeline)

A lightweight, locally deployable GenAI pipeline that performs **fundamental analysis** of industries by analyzing **PDFs, Excel sheets, and API sources** assigned by the user. This project leverages Retrieval-Augmented Generation (RAG) with local LLMs and embedding models.

---

## 🧠 Features

- Ingest data from **PDF**, **Excel**, and **API** sources
- Preprocess and store data as clean text files
- Generate vector embeddings locally with `sentence-transformers`
- Use lightweight **LLMs** such as `phi-3-mini` or `mistral-7b`
- Perform contextual question-answering via **RAG**
- Simple **CLI or Streamlit interface** for interaction

---

## 🧱 Project Structure

```

genai\_fundamental\_analysis/
├── data\_sources/        # Raw input data (PDF, Excel, API)
├── configs/             # Configuration YAMLs
├── embeddings/          # Vector store & intermediate text chunks
├── models/              # Local LLMs if needed
├── scripts/             # Modular pipeline scripts
├── main.py              # Entry point script
└── README.md            # Project documentation

````

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/{os.userid}/genai_fundamental_analysis.git
cd genai_fundamental_analysis

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python main.py
````

---

## 🛠️ Tools & Frameworks

* **LLMs**: `phi-3-mini` / `mistral-7b` (via `llama.cpp`)
* **Embeddings**: `BAAI/bge-small-en` (`sentence-transformers`)
* **Vector DB**: `FAISS` / `Chroma`
* **Frameworks**: `LangChain` or `LlamaIndex` for RAG
* **Interface**: `Streamlit` (optional UI)

---

## ✅ Use Case Example

> “Analyze all uploaded PDFs and Excel files related to the Pharma sector. Summarize key risks, growth indicators, and recent developments.”

The system will:

1. Ingest and parse source files
2. Create vector embeddings
3. Use a local LLM to search and synthesize answers
4. Present results in CLI or Streamlit

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo, raise issues, or submit pull requests.

You can also connect with me on [LinkedIn](https://www.linkedin.com/) to collaborate on similar use cases.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [HuggingFace Transformers](https://huggingface.co/transformers/)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)
* [BAAI bge-small-en](https://huggingface.co/BAAI/bge-small-en)

---

