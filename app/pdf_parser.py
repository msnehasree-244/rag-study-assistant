"""
Extracts raw text per page from an uploaded PDF.

Using pypdf here because it's pure-Python and has zero system
dependencies, which keeps setup simple for a weekend project.
Talking point for interviews: this only handles text-based PDFs.
Scanned/image PDFs would need OCR (e.g. pytesseract) as a fallback --
worth mentioning as a known limitation.
"""
from pypdf import PdfReader


def extract_pages(file_path: str) -> list[str]:
    """Returns a list where element i is the raw text of page i+1."""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages
