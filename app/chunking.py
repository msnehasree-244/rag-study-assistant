"""
Chunking logic: turns raw page text into overlapping chunks.

Why fixed-size + overlap, and not "just split by paragraph"?
- Paragraphs vary wildly in length (a heading is a "paragraph", so is
  three pages of dense text) so paragraph-only splitting gives very
  uneven chunk sizes, which hurts embedding quality.
- Overlap (200 chars here) means a sentence that gets cut at a chunk
  boundary still appears in full in the neighboring chunk, so we don't
  lose meaning at the edges.

Interview-ready upgrade path: semantic chunking, where you embed
sentences and split at points where consecutive-sentence similarity
drops (topic boundary), instead of at a fixed character count.
"""
from dataclasses import dataclass
from app.config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


@dataclass
class Chunk:
    text: str
    page_number: int
    chunk_index: int


def chunk_page_text(text: str, page_number: int, start_index: int = 0) -> list[Chunk]:
    """Split a single page's text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    idx = start_index
    text_len = len(text)

    while start < text_len:
        end = min(start + CHUNK_SIZE_CHARS, text_len)

        # try to break on a sentence/word boundary instead of mid-word
        if end < text_len:
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, page_number=page_number, chunk_index=idx))
            idx += 1

        if end >= text_len:
            break

        # move forward, but overlap with the previous chunk
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)

    return chunks


def chunk_document(pages: list[str]) -> list[Chunk]:
    """pages[i] = raw text of page i+1. Returns all chunks across the doc."""
    all_chunks = []
    running_index = 0
    for page_num, page_text in enumerate(pages, start=1):
        page_chunks = chunk_page_text(page_text, page_num, running_index)
        all_chunks.extend(page_chunks)
        running_index += len(page_chunks)
    return all_chunks
