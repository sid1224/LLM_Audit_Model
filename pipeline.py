"""
pipeline.py
Unified text processing pipeline for Readily Take-Home assignment.

1. Extracts audit questions from the uploaded audit PDF.
2. Extracts text from one or more policy PDFs.
3. Returns structured data ready for LLM-based evaluation.
"""

import logging
from io import BytesIO
from typing import List, Dict, Union

from pdf_parser import extract_multiple_pdfs
from audit_parser import extract_audit_questions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_audit_and_policies(
    audit_pdf: Union[str, BytesIO],
    policy_pdfs: List[Union[str, BytesIO]]
) -> Dict[str, Union[List[str], str]]:
    """
    Full end-to-end PDF processing pipeline.

    Parameters
    ----------
    audit_pdf : str | BytesIO
        PDF file containing audit/compliance questions.
    policy_pdfs : list[str | BytesIO]
        One or more PDFs containing policy text (reference material).

    Returns
    -------
    dict
        {
            "questions": [list of extracted audit questions],
            "policy_text": "combined text from all uploaded policy PDFs"
        }
    """

    logger.info("Starting Readily pipeline...")

    # --- Step 1: Extract questions from audit PDF ---
    logger.info("Extracting audit questions...")
    questions = extract_audit_questions(audit_pdf)
    logger.info(f"✅ Found {len(questions)} questions.")

    # --- Step 2: Extract text from policy PDFs ---
    logger.info("Extracting text from policy PDFs...")
    policy_text = extract_multiple_pdfs(policy_pdfs)
    logger.info(f"✅ Extracted {len(policy_text)} characters of policy text.")

    # --- Step 3: Return structured result ---
    result = {
        "questions": questions,
        "policy_text": policy_text
    }

    logger.info("Pipeline complete.")
    return result


# -------------------------------------------------------------------------
# Example local test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    audit_path = r"C:\Users\disis\LLM_Audit_Model\Readily Take Home (ENG).pdf"
    policy_paths = [
        r"C:\Users\disis\LLM_Audit_Model\AA.1000_CEO20250206_v20250201.pdf",
        r"C:\Users\disis\LLM_Audit_Model\GG.1508_CEO20241122_v20241101.pdf"
    ]

    # open each policy file and collect file objects
    with open(audit_path, "rb") as a_file:
        policy_files = [open(p, "rb") for p in policy_paths]
        try:
            output = process_audit_and_policies(a_file, policy_files)
        finally:
            # always close opened policy files
            for pf in policy_files:
                pf.close()

    print(f"\nExtracted {len(output['questions'])} questions:")
    for i, q in enumerate(output['questions'], 1):
        print(f"{i}. {q}")

    print(f"\nTotal policy text length: {len(output['policy_text'])} characters")