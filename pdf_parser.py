import io
import re
import logging
from typing import Union, List, Tuple, Dict

from PyPDF2 import PdfReader
import pdfplumber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    text = text.replace('\x00', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_pypdf2(file: Union[str, io.BytesIO]) -> List[Tuple[str, int]]:
    """
    Returns list of tuples: (page_text, page_number)
    """
    try:
        reader = PdfReader(file)
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages_text.append((clean_text(text), i + 1))  # Page numbers start at 1
        return pages_text
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}")
        return []


def extract_text_pdfplumber(file: Union[str, io.BytesIO]) -> List[Tuple[str, int]]:
    """
    Returns list of tuples: (page_text, page_number)
    """
    try:
        pages_text = []
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                pages_text.append((clean_text(text), i + 1))
        return pages_text
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
        return []


def extract_pdf_text_with_pages(file: Union[str, io.BytesIO]) -> List[Tuple[str, int]]:
    """
    Unified extraction: tries PyPDF2 first, falls back to pdfplumber.
    Returns list of (page_text, page_number)
    """
    pages_text = extract_text_pypdf2(file)
    if not pages_text or sum(len(t[0]) for t in pages_text) < 500:
        logger.info("PyPDF2 extracted very little text; falling back to pdfplumber.")
        pages_text = extract_text_pdfplumber(file)
    return pages_text


# --- ORIGINAL FUNCTION ---
def extract_multiple_pdfs_with_pages(files: List[Union[str, io.BytesIO]]) -> List[Tuple[str, int]]:
    """
    Returns a combined list of tuples: (page_text, page_number)
    """
    combined_pages = []
    for f in files:
        pdf_name = getattr(f, "name", "uploaded_file") 
        logger.info(f"Extracting text from {pdf_name}")
        pages = extract_pdf_text_with_pages(f)
        # 🚨 FIX 1: Include the pdf_name in the page data 🚨
        for page_text, page_num in pages:
            combined_pages.append((page_text, page_num, pdf_name)) # Now a 3-tuple
    return combined_pages

def extract_sections_with_headings(page_text: str, page_number: int) -> List[Dict]:
    """
    Splits a page's text into sections based on headings (e.g., "A. Section Title").
    Returns a list of dicts: {'heading', 'text', 'page_number'}
    """
    pattern = r'([A-Z]\.\s+[A-Za-z ].+)'  # simple heading pattern
    matches = list(re.finditer(pattern, page_text))
    sections = []

    if not matches:
        return [{"heading": "No Heading", "text": page_text.strip(), "page_number": page_number}]

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        heading = m.group(1).strip()
        section_text = page_text[start:end].strip()
        sections.append({
            "heading": heading,
            "text": section_text,
            "page_number": page_number
        })
    return sections

