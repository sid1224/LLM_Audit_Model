"""
audit_parser.py
Comprehensive audit question extractor for Readily Take-Home.
Extracts all question-like sentences with high precision and recall.
"""

import re
from typing import List, Union, IO
from pdf_parser import extract_pdf_text


def extract_audit_questions(file: Union[str, IO[bytes]]) -> List[str]:
    """
    Extracts clean, complete audit questions from the given PDF.
    Logic:
    - Split into sentences.
    - Keep any sentence that ends with '?'.
    - Prefer those starting with interrogative or modal words.
    """

    # 1. Extract text and normalise spacing
    text = extract_pdf_text(file)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Split into sentences by punctuation
    sentences = re.split(r'(?<=[.?!])\s+', text)

    # 3. Expanded list of question starters (case-insensitive)
    question_starters = (
        "does", "do", "is", "are", "was", "were",
        "will", "would", "should", "shall", "can", "could",
        "may", "might", "must", "has", "have", "had",
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "explain", "state", "define", "describe", "outline"
    )

    questions = []

    for s in sentences:
        s = s.strip()
        if not s or len(s) < 8:
            continue

        s_lower = s.lower().lstrip("0123456789). ")

        # Condition 1: Sentence starts with question-like word AND ends with '?'
        starts_like_question = any(s_lower.startswith(w) for w in question_starters)
        ends_with_qmark = s.endswith("?")

        # Condition 2: Even if it doesn’t start with a question word,
        # still include if it ends with '?'
        if ends_with_qmark:
            # Clean stray punctuation before '?'
            s = re.sub(r'[^A-Za-z0-9,;:\-()\'\"/& ]+\?$', '?', s)
            # Add if it starts like a question OR just ends with '?'
            if starts_like_question or ends_with_qmark:
                questions.append(s)

    # 4. Deduplicate while preserving order
    seen, unique_questions = set(), []
    for q in questions:
        if q not in seen:
            unique_questions.append(q)
            seen.add(q)

    return unique_questions


file_path = r"C:\Users\disis\LLM_Audit_Model\Readily Take Home (ENG).pdf"

if __name__ == "__main__":
    with open(file_path, "rb") as f:
        qs = extract_audit_questions(f)

    print(f"Extracted {len(qs)} questions:")
    for i, q in enumerate(qs, 1):
        print(f"{i}. {q}")


