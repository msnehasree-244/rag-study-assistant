"""
Takes retrieved chunks + the user's question, builds a grounded prompt,
and calls Gemini. The prompt is deliberately strict about staying
inside the retrieved context and citing pages -- this is the piece
that stops "RAG" from just being "an API call with extra steps."
"""
from google import genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


SYSTEM_PROMPT = """You are a study assistant that answers questions ONLY using \
the provided excerpts from the student's own notes/textbook. Rules:

1. Only use information present in the excerpts below. Do not use outside knowledge.
2. Every claim you make must be traceable to a specific excerpt. Reference the \
page number in brackets, e.g. [p. 4], right after the relevant sentence.
3. If the excerpts don't contain enough information to answer the question, \
say so plainly instead of guessing or filling gaps with general knowledge.
4. Be concise and direct -- this is for exam prep, not an essay."""


def build_context_block(hits: list[dict]) -> str:
    """Turns retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, hit in enumerate(hits, start=1):
        page = hit["metadata"]["page_number"]
        parts.append(f"[Excerpt {i}, page {page}]\n{hit['text']}")
    return "\n\n".join(parts)


def estimate_confidence(hits: list[dict]) -> str:
    """Rough confidence signal from retrieval distance, so the UI can flag
    low-confidence answers instead of presenting everything as equally sure.
    Cosine distance: lower = more similar. Thresholds are tuned loosely --
    a real system would calibrate these against labeled examples."""
    if not hits:
        return "low"
    best_distance = min(h["distance"] for h in hits)
    if best_distance < 0.35:
        return "high"
    elif best_distance < 0.6:
        return "medium"
    return "low"


def answer_question(question: str, hits: list[dict]) -> dict:
    if not hits:
        return {
            "answer": "I couldn't find anything relevant to that question in your uploaded document.",
            "confidence": "low",
            "sources": [],
        }

    context_block = build_context_block(hits)
    user_message = f"Excerpts from the document:\n\n{context_block}\n\nQuestion: {question}"

    client = get_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{SYSTEM_PROMPT}\n\n{user_message}",
    )

    answer_text = response.text

    sources = [
        {
            "page": hit["metadata"]["page_number"],
            "document_name": hit["metadata"]["document_name"],
            "excerpt": hit["text"][:200] + ("..." if len(hit["text"]) > 200 else ""),
        }
        for hit in hits
    ]

    return {
        "answer": answer_text,
        "confidence": estimate_confidence(hits),
        "sources": sources,
    }
