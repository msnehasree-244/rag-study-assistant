"""
Takes retrieved chunks + the user's question, builds a prompt, and calls
Gemini. Supports two modes:

- Grounded mode (default): answers ONLY from the retrieved excerpts, refuses
  to answer if the document doesn't contain the info. This is the "no
  hallucination" mode -- the whole point of RAG.
- General knowledge mode: still shows the retrieved excerpts as context if
  relevant, but allows the model to also use its own knowledge to fully
  answer the question (e.g. solving problems from a question bank that has
  no answers in the document). The answer is labeled so the student knows
  what came from their notes vs. the model's own knowledge.
"""
from google import genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


GROUNDED_SYSTEM_PROMPT = """You are a study assistant that answers questions ONLY using \
the provided excerpts from the student's own notes/textbook. Rules:

1. Only use information present in the excerpts below. Do not use outside knowledge.
2. Every claim you make must be traceable to a specific excerpt. Reference the \
page number in brackets, e.g. [p. 4], right after the relevant sentence.
3. If the excerpts don't contain enough information to answer the question, \
say so plainly instead of guessing or filling gaps with general knowledge.
4. Be concise and direct -- this is for exam prep, not an essay."""

GENERAL_KNOWLEDGE_SYSTEM_PROMPT = """You are a study assistant helping a student \
with their notes/textbook. You have two sources of information: excerpts from \
their document (below), and your own general knowledge. Rules:

1. If the excerpts below answer the question, prioritize them and cite the page \
number in brackets, e.g. [p. 4].
2. If the excerpts don't fully answer the question (e.g. the document only has \
questions with no answers, or is missing information), use your own knowledge \
to give a complete, correct answer -- but clearly label which parts came from \
the student's document versus your own knowledge, e.g. by starting a section \
with "From your notes:" vs "From general knowledge:".
3. Be concise and direct -- this is for exam prep, not an essay."""


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


def answer_question(question: str, hits: list[dict], use_general_knowledge: bool = False) -> dict:
    # In grounded mode, no hits means we truly have nothing to work with.
    # In general knowledge mode, we can still answer using the model's own
    # knowledge even with zero relevant hits, so don't early-return there.
    if not hits and not use_general_knowledge:
        return {
            "answer": "I couldn't find anything relevant to that question in your uploaded document.",
            "confidence": "low",
            "sources": [],
            "mode": "grounded",
        }

    context_block = build_context_block(hits) if hits else "(No relevant excerpts found in the document.)"
    user_message = f"Excerpts from the document:\n\n{context_block}\n\nQuestion: {question}"

    system_prompt = GENERAL_KNOWLEDGE_SYSTEM_PROMPT if use_general_knowledge else GROUNDED_SYSTEM_PROMPT

    client = get_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{system_prompt}\n\n{user_message}",
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
        "confidence": estimate_confidence(hits) if hits else "low",
        "sources": sources,
        "mode": "general_knowledge" if use_general_knowledge else "grounded",
    }
