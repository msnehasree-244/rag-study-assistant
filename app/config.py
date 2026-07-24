"""
Central place for all config. Keeping this separate means you can
explain in an interview: "config is injected, not hardcoded, so this
would work the same way in prod with different env vars."
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# --- Embeddings ---
# Local, free, no API cost. 384-dim vectors, good enough for a student project.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Chunking ---
# Fixed-size chunking with overlap. Simple, explainable, works fine for v1.
# Interview talking point: "next step would be semantic chunking using
# sentence boundaries / embedding similarity to decide split points."
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "5"))

# --- Storage ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
