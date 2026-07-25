"""
FastAPI app: two endpoints.

POST /documents/upload  -> ingest a PDF (parse, chunk, embed, store)
POST /query              -> ask a question, get a grounded answer + citations
GET  /documents          -> list uploaded documents (for the frontend dropdown)

Note on async processing: for a weekend project, ingestion runs
synchronously inside the request. That's fine for small PDFs. The
documented next step (say this in interviews) is to push ingestion
onto a background job queue (e.g. Celery + Redis, or FastAPI
BackgroundTasks as a lighter first step) so a big PDF upload doesn't
block the request/timeout the client.
"""
import os
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import UPLOAD_DIR
from app.pdf_parser import extract_pages
from app.chunking import chunk_document
from app.vectorstore import add_chunks, query as vector_query, get_collection
from app.generation import answer_question
from app.quiz import generate_quiz, generate_flashcards

app = FastAPI(title="Chat With Your Notes - RAG API")

# Loosen CORS for local dev / Lovable frontend calling this API.
# Tighten this to your actual frontend origin before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory registry of uploaded docs (name/id). For a weekend project
# this is fine; swap for a real DB table if you want it to survive restarts
# independent of Chroma's metadata.
_documents: dict[str, str] = {}


class QueryRequest(BaseModel):
    question: str
    document_id: str | None = None  # omit to search across all documents
    use_general_knowledge: bool = False  # True = allow answers beyond the document


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported right now.")

    document_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    pages = extract_pages(save_path)
    if not any(p.strip() for p in pages):
        raise HTTPException(
            422,
            "Couldn't extract any text from this PDF. It may be a scanned "
            "image PDF, which needs OCR support (not implemented yet).",
        )

    chunks = chunk_document(pages)
    stored_count = add_chunks(document_id, file.filename, chunks)

    _documents[document_id] = file.filename

    return {
        "document_id": document_id,
        "document_name": file.filename,
        "page_count": len(pages),
        "chunk_count": stored_count,
    }


@app.get("/documents")
def list_documents():
    return [{"document_id": k, "document_name": v} for k, v in _documents.items()]


@app.post("/query")
def query_document(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    hits = vector_query(req.question, document_id=req.document_id)
    result = answer_question(req.question, hits, use_general_knowledge=req.use_general_knowledge)
    return result


class QuizRequest(BaseModel):
    document_id: str
    count: int = 5


class FlashcardRequest(BaseModel):
    document_id: str
    count: int = 10


@app.post("/quiz")
def create_quiz(req: QuizRequest):
    if req.document_id not in _documents:
        raise HTTPException(404, "Document not found. Upload it first.")
    return generate_quiz(req.document_id, req.count)


@app.post("/flashcards")
def create_flashcards(req: FlashcardRequest):
    if req.document_id not in _documents:
        raise HTTPException(404, "Document not found. Upload it first.")
    return generate_flashcards(req.document_id, req.count)
