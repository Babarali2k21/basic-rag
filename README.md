# DocChat (Production RAG Pipeline)

[![CI](https://github.com/Babarali2k21/basic-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/Babarali2k21/basic-rag/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/Babarali2k21/basic-rag/graph/badge.svg?token=CDDQYTLPZ9)](https://codecov.io/github/Babarali2k21/basic-rag)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6F00)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![RAGAS](https://img.shields.io/badge/Evaluated-RAGAS-7C3AED)](https://docs.ragas.io)

> Upload `.txt` or `.pdf` documents, ask questions in plain English.

**[Live Demo → basic-rag-mstghx3dbkhezdategemlv.streamlit.app](https://basic-rag-mstghx3dbkhezdategemlv.streamlit.app)**

---


## Evaluation Results (RAGAS)

Evaluated on a QA dataset over the included articles (`python evaluate.py`):

| Metric | Score | What it means |
|---|---|---|
| **Faithfulness** | 0.92 | Answers grounded in retrieved context |
| **Answer Relevancy** | 0.88 | Answers address the question asked |
| **Context Precision** | 0.85 | Retrieved chunks are on-topic |
| **Context Recall** | 0.81 | Necessary chunks were retrieved |

> Run `python evaluate.py` to reproduce with your own documents.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────┐
│       Streamlit UI  /  CLI (main.py)    │
└──────────────────┬──────────────────────┘
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
┌──────────────┐      ┌──────────────────────┐
│ data_loader  │      │    vector_store       │
│              │      │                       │
│ .txt + .pdf  ├─────▶│  OpenAI Embeddings    │
│ LangChain    │chunks│  ChromaDB (persisted) │
│ TextSplitter │      │  Similarity search    │
└──────────────┘      └──────────┬───────────┘
                                 │  top-k chunks
                                 ▼
                      ┌──────────────────────┐
                      │       rag.py          │
                      │                       │
                      │  LangChain LCEL chain │
                      │  GPT-4o-mini          │
                      │  Source citations     │
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │     RAGResponse       │
                      │  answer: str          │
                      │  sources: list[str]   │
                      │  num_chunks_used: int │
                      └──────────────────────┘
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Babarali2k21/basic-rag.git
cd basic-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here
```

### 3. Add your documents

Drop `.txt` or `.pdf` files into the `./articles/` folder. A sample article is included to get you started.

### 4a. Web UI (recommended)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), enter your API key in the sidebar, upload documents or use the articles folder, then start asking questions.

### 4b. CLI

```bash
# Index documents
python main.py --index

# Interactive Q&A session
python main.py

# Single question
python main.py --query "What is retrieval-augmented generation?"
```

### 5. Run evaluation (optional)

```bash
python evaluate.py
```

---

## Project Structure

```
basic-rag/
├── app.py              # Streamlit web UI with document upload
├── main.py             # CLI (--index / --query / interactive)
├── config.py           # Settings via Pydantic BaseSettings
├── data_loader.py      # Load .txt/.pdf + LangChain text splitting
├── vector_store.py     # ChromaDB indexing, loading, retriever factory
├── rag.py              # LangChain LCEL chain + RAGResponse dataclass
├── evaluate.py         # RAGAS evaluation script
├── conftest.py         # pytest path configuration
├── articles/           # Drop knowledge base files here
│   └── rag_overview.txt
├── docs/
│   └── screenshot.png  # App screenshot
├── tests/
│   └── test_rag.py     # Unit + integration tests (mocked, no API key needed)
├── .env.example        # Environment variable template
├── requirements.txt
└── .github/
    └── workflows/
        └── tests.yml   # CI: pytest + codecov on Python 3.11 & 3.12
```

---

## Configuration

All settings via `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM for answer generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `RETRIEVAL_K` | `5` | Chunks to retrieve per query |
| `CHROMA_PATH` | `chroma_persistent_storage` | ChromaDB persistence path |
| `ARTICLES_DIR` | `./articles` | Knowledge base directory |

---

## Tests

```bash
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=term-missing
```

All tests use mocked LLM calls **no API key required** to run the test suite.

**12/12 tests passing on Python 3.11 and 3.12.**

---

## Roadmap

- [x] Core RAG pipeline (load → chunk → embed → retrieve → generate)
- [x] PDF support via PyPDF
- [x] Document upload via Streamlit UI
- [x] LangChain LCEL chain with source citations
- [x] Streamlit web UI with chat history
- [x] RAGAS evaluation script
- [x] Pydantic v2 settings
- [x] 12 pytest tests (mocked, no API key needed)
- [x] CI/CD (GitHub Actions + Codecov)
- [x] Deployed on Streamlit Cloud
- [ ] Conversational memory (multi-turn Q&A)
- [ ] Advanced retrieval: HyDE + cross-encoder reranking
- [ ] Pinecone / Qdrant as vector store alternatives
- [ ] Docker deployment

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| RAG Framework | LangChain 0.2 |
| Vector Store | ChromaDB |
| Evaluation | RAGAS |
| Web UI | Streamlit |
| Config | Pydantic BaseSettings |
| Testing | pytest + pytest-cov (12 tests) |
| CI/CD | GitHub Actions + Codecov |
| Deployment | Streamlit Cloud |

---

## Author

**Babar Ali**  AI Engineer · Vienna, Austria

[![LinkedIn](https://img.shields.io/badge/LinkedIn-babarali2k21-0A66C2?logo=linkedin)](https://linkedin.com/in/babarali2k21)
[![GitHub](https://img.shields.io/badge/GitHub-Babarali2k21-181717?logo=github)](https://github.com/Babarali2k21)

---

## License

MIT
