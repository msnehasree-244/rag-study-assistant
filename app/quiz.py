"""
Generates quizzes and flashcards from a document's content.

Reuses the exact same retrieval pipeline as Q&A -- pull relevant
chunks from Chroma, then ask Gemini to produce structured
questions/flashcards grounded in those chunks, returned as clean JSON
the frontend can render directly (no parsing markdown on the client).
"""
import json
from google import genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.vectorstore import get_collection

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def get_all_chunks_for_document(document_id: str, limit: int = 20) -> list[dict]:
    """Pull a sample of chunks for a document directly from Chroma
    (not similarity search -- we want broad coverage of the document
    for quiz generation, not chunks relevant to one specific question)."""
    collection = get_collection()
    results = collection.get(
        where={"document_id": document_id},
        limit=limit,
    )
    chunks = []
    if results["ids"]:
        for i in range(len(results["ids"])):
            chunks.append({
                "text": results["documents"][i],
                "page": results["metadatas"][i]["page_number"],
            })
    return chunks


QUIZ_PROMPT_TEMPLATE = """Based ONLY on the following excerpts from a student's \
document, generate {count} multiple-choice quiz questions to help them study.

Rules:
- Every question must be answerable directly from the excerpts below.
- Provide exactly 4 options per question, only one correct.
- Include the page number the answer comes from.
- Do not invent facts not present in the excerpts.

Respond with ONLY valid JSON (no markdown fences, no preamble), in this exact shape:
{{
  "questions": [
    {{
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_answer_index": 0,
      "page": 1,
      "explanation": "brief reason why this is correct, grounded in the excerpt"
    }}
  ]
}}

Excerpts:
{context}
"""

FLASHCARD_PROMPT_TEMPLATE = """Based ONLY on the following excerpts from a student's \
document, generate {count} flashcards to help them memorize key concepts.

Rules:
- Front of card: a term or short question.
- Back of card: a concise definition or answer, grounded in the excerpts.
- Include the page number as reference.
- Do not invent facts not present in the excerpts.

Respond with ONLY valid JSON (no markdown fences, no preamble), in this exact shape:
{{
  "flashcards": [
    {{
      "front": "...",
      "back": "...",
      "page": 1
    }}
  ]
}}

Excerpts:
{context}
"""


def _build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[page {c['page']}]\n{c['text']}" for c in chunks)


def _call_gemini_for_json(prompt: str) -> dict:
    client = get_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    raw_text = response.text.strip()

    # Gemini sometimes wraps JSON in markdown fences despite instructions --
    # strip them defensively rather than trusting the prompt alone.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    return json.loads(raw_text)


def generate_quiz(document_id: str, count: int = 5) -> dict:
    chunks = get_all_chunks_for_document(document_id)
    if not chunks:
        return {"questions": []}

    context = _build_context(chunks)
    prompt = QUIZ_PROMPT_TEMPLATE.format(count=count, context=context)
    return _call_gemini_for_json(prompt)


def generate_flashcards(document_id: str, count: int = 10) -> dict:
    chunks = get_all_chunks_for_document(document_id)
    if not chunks:
        return {"flashcards": []}

    context = _build_context(chunks)
    prompt = FLASHCARD_PROMPT_TEMPLATE.format(count=count, context=context)
    return _call_gemini_for_json(prompt)
