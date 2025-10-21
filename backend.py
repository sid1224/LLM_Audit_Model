import os
import io
import tempfile
import asyncio
import pickle
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber

# Import your Gemini evaluator functions
from llm_compliance_evaluator import (
    evaluate_with_semantic_search,
    split_into_paragraphs,
    EMBEDDING_CACHE_FILE
)

# Load API key from environment
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set!")

# FastAPI setup
app = FastAPI(title="Compliance Evaluator API")


# --------------------
# Helper function: Extract text from PDF bytes
# --------------------
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# --------------------
# Endpoint for evaluation
# --------------------
@app.post("/evaluate")
async def evaluate(
    policy_files: List[UploadFile] = File(...),
    questions_file: UploadFile = File(...)
):
    try:
        # 1️⃣ Extract questions from the uploaded PDF
        questions_bytes = await questions_file.read()
        questions_text = extract_text_from_pdf_bytes(questions_bytes)
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and len(q.strip()) > 10]
        if not questions:
            raise HTTPException(status_code=400, detail="No valid questions extracted from PDF.")

        # 2️⃣ Save uploaded policy PDFs to temporary files
        pdf_paths = []
        temp_files = []
        for f in policy_files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(await f.read())
            tmp.close()
            pdf_paths.append(tmp.name)
            temp_files.append(tmp.name)  # keep track to delete later

        # 3️⃣ Run evaluation with Gemini RAG
        results = await evaluate_with_semantic_search(API_KEY, questions, pdf_paths)

        # 4️⃣ Cleanup temporary files
        for tmp_file in temp_files:
            try:
                os.remove(tmp_file)
            except Exception:
                pass  # ignore cleanup errors

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------
# Local testing
# --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)