# RAG Chunking Playground

A Python backend for experimenting with **Retrieval-Augmented Generation (RAG)** over PDF documents. Upload a PDF, choose a chunking strategy, store embeddings in Qdrant, and ask questions answered using retrieved document context.

Built as a learning project with **FastAPI, LangChain, Sentence Transformers, Qdrant (Local Mode), and Groq**.

## Features

- Upload PDFs and extract text with PyMuPDF.
- Preview chunks before indexing.
- Choose from six chunking strategies: **fixed-size, recursive, sentence, paragraph, semantic, and sliding window**.
- Configure chunk size, overlap, Top-K retrieval, and LLM output tokens.
- Generate local embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Store vectors and chunk metadata persistently in **Qdrant Local Mode** (no Docker required).
- Create multiple indexes for the same PDF to compare chunking approaches.
- Ask PDF questions with **Groq** and inspect the retrieved source chunks and page numbers.

## Tech Stack

| Component | Technology |
| --- | --- |
| API | FastAPI |
| PDF extraction | PyMuPDF |
| RAG integrations and text splitting | LangChain |
| Embeddings | Sentence Transformers (Hugging Face) |
| Vector database | Qdrant Local Mode |
| Answer generation | Groq |
| Testing | pytest |

## Project Structure

```text
rag-chunking-project/
├── app/
│   ├── api/
│   │   ├── routes.py           # PDF upload and document APIs
│   │   ├── chunk_routes.py     # Chunk preview API
│   │   └── rag_routes.py       # Index, retrieval, and chat APIs
│   ├── chunkers/               # Chunking strategies and factory
│   ├── services/
│   │   ├── document_store.py
│   │   ├── pdf_service.py
│   │   ├── embedding_service.py
│   │   ├── vector_service.py
│   │   ├── index_store.py
│   │   └── rag_service.py
│   ├── config.py
│   ├── schemas.py
│   └── main.py
├── tests/                      # Local tests (Git-ignored)
├── data/                       # Local documents, indexes, Qdrant (Git-ignored)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd rag-chunking-project
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Set your Groq API key in `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DATA_DIR=data/documents
INDEX_DIR=data/indexes
QDRANT_PATH=data/qdrant
```

**Never commit `.env` or your real API key.** The first use of the embedding model may download its weights.

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

Open the interactive Swagger documentation at **http://127.0.0.1:8000/docs**.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/documents/upload` | Upload and extract a PDF |
| `POST` | `/api/v1/chunks/preview` | Preview chunks for selected settings |
| `POST` | `/api/v1/indexes` | Chunk a document and build a Qdrant index |
| `GET` | `/api/v1/indexes` | List existing indexes |
| `GET` | `/api/v1/indexes/{index_id}` | View index metadata |
| `GET` | `/api/v1/indexes/{index_id}/chunks` | Inspect stored chunks |
| `DELETE` | `/api/v1/indexes/{index_id}` | Delete an index |
| `POST` | `/api/v1/retrieval/search` | Retrieve relevant PDF chunks |
| `POST` | `/api/v1/chat` | Generate an answer with sources |

## How It Works

**Indexing**

```text
Upload PDF → Extract text → Select chunking strategy
           → Generate chunks → Embed chunks → Store in Qdrant
```

**Question answering**

```text
User question → Embed question → Qdrant similarity search
              → Build context from retrieved chunks → Groq
              → Return answer and source metadata
```

### Quick Test in Swagger

1. Use `POST /api/v1/documents/upload` to upload a PDF. Copy the returned `document_id`.
2. Use `POST /api/v1/indexes` with that ID and the desired settings. Copy the returned `index_id`.
3. Use `POST /api/v1/retrieval/search` to inspect the retrieved chunks.
4. Use `POST /api/v1/chat` to ask questions about the PDF.

Example indexing request:

```json
{
  "document_id": "YOUR-DOCUMENT-UUID",
  "settings": {
    "technique": "paragraph",
    "chunk_size": 400,
    "chunk_overlap": 0,
    "top_k": 4,
    "max_output_tokens": 512
  }
}
```

Example chat request:

```json
{
  "index_id": "YOUR-INDEX-UUID",
  "question": "What are the main concepts discussed in this PDF?",
  "top_k": 4,
  "max_output_tokens": 512
}
```

**Note:** `document_id` identifies an uploaded PDF; `index_id` identifies one chunking/indexing experiment. You can create multiple indexes for a single PDF.

## Testing

Run the local test suite:

```bash
python -m pytest tests/ -v
```

The backend has passed **31 tests**, including API integration tests using in-memory Qdrant, fake embeddings, and a mocked LLM. The complete flow was also manually verified with a real PDF, local embeddings, Qdrant, and Groq.

The `tests/` directory is intentionally excluded from Git for this project.

## Notes

- Uploaded documents, vector data, and index metadata are stored locally under `data/` and excluded from Git.
- Groq is called to generate answers; local Qdrant handles similarity retrieval.
- Retrieved chunks are supporting evidence, but retrieval relevance and model citations should still be evaluated.
- The **frontend will be developed separately** with Next.js.
