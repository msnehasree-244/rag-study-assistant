"""
Thin wrapper around ChromaDB + sentence-transformers.

Why Chroma: it's embedded (no separate server to run/deploy), persists
to disk, and is free -- ideal for a self-hosted student project where
you want to say "I understand what's happening under the hood" rather
than "I called Pinecone's API."

Why sentence-transformers (all-MiniLM-L6-v2): runs fully locally, no
API cost or key needed, 384-dim embeddings, good enough quality for a
project this size, and fast on CPU.
"""
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME, TOP_K

_embedding_model = None
_chroma_client = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_collection():
    """Single collection holds chunks from ALL documents.
    Each chunk is tagged with document_id in its metadata, so retrieval
    can be scoped per-document or run across everything (multi-doc)."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name="study_notes",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_chunks(document_id: str, document_name: str, chunks: list) -> int:
    """Embed and store chunks for one document. Returns number stored."""
    if not chunks:
        return 0

    model = get_embedding_model()
    collection = get_collection()

    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    ids = [f"{document_id}_{c.chunk_index}" for c in chunks]
    metadatas = [
        {
            "document_id": document_id,
            "document_name": document_name,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


def query(question: str, document_id: str | None = None, top_k: int = TOP_K) -> list[dict]:
    """Retrieve the top_k most relevant chunks for a question.
    If document_id is given, search is scoped to that document only.
    Omitting it enables multi-document retrieval."""
    model = get_embedding_model()
    collection = get_collection()

    query_embedding = model.encode([question]).tolist()

    where_filter = {"document_id": document_id} if document_id else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
    )

    hits = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                # Chroma returns distance; smaller = more similar for cosine space.
                "distance": results["distances"][0][i],
            })
    return hits
