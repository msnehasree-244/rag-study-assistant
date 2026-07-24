# Chat With Your Notes — RAG Backend

A minimal, self-hosted RAG (Retrieval-Augmented Generation) pipeline:
upload a PDF, ask questions, get answers grounded in the document with
page citations.

## Architecture

```
PDF upload → pypdf (extract text per page)
           → chunking.py (fixed-size chunks, 1200 chars, 200 overlap)
           → sentence-transformers (all-MiniLM-L6-v2, local embeddings)
           → ChromaDB (persisted locally, self-hosted vector store)

Question → embed question → Chroma similarity search (top-5 chunks)
         → build grounded prompt with citations
         → Claude (Anthropic API) → answer + confidence + sources
```

## Setup

```bash
cd rag-backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-key-here
```

(Get a key from https://console.anthropic.com — the sentence-transformers
embedding model needs NO API key, it runs fully locally.)

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

First run will download the embedding model (~80MB) automatically —
that's normal, just takes a minute.

API docs (auto-generated, useful for testing without a frontend yet):
http://localhost:8000/docs

## Endpoints

- `POST /documents/upload` — multipart form upload, field name `file`, PDF only
- `GET /documents` — list uploaded docs (for a dropdown in your frontend)
- `POST /query` — JSON body `{"question": "...", "document_id": "optional"}`

## Connecting from Lovable

Point your Lovable frontend's API calls at wherever you deploy this
(e.g. Railway, Render, Fly.io — all have free tiers). CORS is wide open
in `main.py` for now (`allow_origins=["*"]`) — tighten that to your
actual frontend domain before you consider this "done."

## Known limitations (be upfront about these in interviews — it shows maturity)

- **Scanned/image PDFs won't work.** pypdf only extracts real text layers.
  Fix: add OCR fallback (pytesseract) when extraction returns empty text.
- **Ingestion is synchronous.** A large PDF blocks the upload request.
  Fix: move ingestion to a background job queue (Celery+Redis, or
  FastAPI `BackgroundTasks` as a lighter first step) so upload returns
  immediately and processing happens async.
- **Chunking is fixed-size, not semantic.** Simple and explainable, but
  can split a sentence or idea across chunk boundaries. Fix: semantic
  chunking — embed sentences, split where consecutive-sentence
  similarity drops (topic boundary).
- **No reranking step.** Retrieval returns raw top-k by cosine
  similarity. Fix: add a cross-encoder reranker (e.g.
  `sentence-transformers` cross-encoder models) after retrieval, before
  sending chunks to the LLM — improves precision noticeably.
- **Confidence scoring is a heuristic**, not a calibrated model. It
  thresholds on retrieval distance. Good enough to flag "we're not
  sure" but worth naming as a simplification.
- **Document registry is in-memory** (`_documents` dict in `main.py`)
  — resets on server restart. Fine for a demo, swap for a real DB table
  (Postgres/Supabase, which Lovable already gives you) for anything
  persistent.

## Talking points for interviews

- **Why fixed-size chunking with overlap?** Predictable, cheap, and the
  overlap prevents losing meaning at chunk boundaries. Trade-off: can
  still split a coherent idea across two chunks — semantic chunking is
  the fix, and you understand why.
- **Why Chroma over Pinecone?** Self-hosted, free, and forces you to
  understand what a vector DB actually does (HNSW indexing, cosine
  similarity) rather than treating it as a managed black box.
- **Why local embeddings over OpenAI's?** No API cost during
  development, no rate limits while iterating, and it's a fair
  trade-off to discuss: OpenAI's `text-embedding-3` models are higher
  quality but paid.
- **How would you evaluate retrieval quality?** Build a small labeled
  set of (question, correct source chunk) pairs, measure precision@k /
  recall@k on retrieval before even looking at the LLM's final answer.
  This separates "is my retrieval good" from "is my prompt good" —
  a distinction most people building RAG apps skip.
- **How would you scale to millions of documents?** Chroma's
  single-node HNSW index won't scale that far — you'd move to a
  distributed vector DB (Pinecone, Weaviate, Qdrant Cloud) or shard by
  user/tenant, add caching for repeated queries, and move ingestion
  fully off the request path into a queue.
