"""
Gemini-only Compliance Evaluator (Updated, Paragraph-Level, List-Aware, Fast)
-------------------------------------------------------------------------------
- Paragraph/subsection-level chunking to avoid missing details
- Uses current Google Gemini embedding & generation endpoints
- List-aware prompt for enumerated points
- Evidence includes PDF name & page number
- Embedding caching for speed
"""

import os
import re
import json
import asyncio
import aiohttp
import pickle
import logging
from typing import List, Dict
import numpy as np

from pdf_parser import extract_multiple_pdfs_with_pages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_CACHE_FILE = "embeddings_cache.pkl"

# ======================
# 1️⃣ Utilities
# ======================
def clean_text(text: str) -> str:
    text = text.replace('\x00', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_paragraphs(text: str, page_number: int, pdf_name: str) -> List[Dict]:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return [{"text": p, "page_number": page_number, "pdf_name": pdf_name} for p in paragraphs]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_relevant_sentences(text: str, question: str, max_sentences: int = 3) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    keywords = set(re.findall(r'\w+', question.lower()))
    relevant = [s for s in sentences if any(k in s.lower() for k in keywords)]
    if not relevant:
        relevant = sentences
    return ' '.join(relevant[:max_sentences])

# ======================
# 2️⃣ Gemini API calls (Updated)
# ======================
# ======================
# 2️⃣ Gemini API calls (Updated)
# ======================
async def get_gemini_embedding(session: aiohttp.ClientSession, api_key: str, text: str,
                               model: str = "embedding-001") -> np.ndarray: # ⬅️ Ensure this is "embedding-001"
    
    # 🚨 CRITICAL FIX: Use the correct, current endpoint path and method 🚨
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    
    # 🚨 CRITICAL FIX: Use the correct, modern request body structure 🚨
    body = {
        "model": model,
        "content": {"parts": [{"text": text}]},
    }
    
    async with session.post(url, headers=headers, json=body, timeout=30) as resp:
        if resp.status != 200:
            text_resp = await resp.text()
            raise RuntimeError(f"Gemini embedding failed: {resp.status} - {text_resp}")
        
        data = await resp.json()
        
        # 🚨 CRITICAL FIX: The result is in a new structure 🚨
        return np.array(data["embedding"]["values"])

# ======================
# 2️⃣ Gemini API calls (Updated for generation)
# ======================
# ======================
# 2️⃣ Gemini API calls (Final Fix for Generation)
# ======================
# ======================
# 2️⃣ Gemini API calls (Final JSON Payload Fix)
# ======================
async def call_gemini_async(session: aiohttp.ClientSession, api_key: str, model: str, prompt: str) -> str:
    
    cleaned_model = model.replace('models/', '')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cleaned_model}:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    
    # 🚨 CRITICAL FIX: Change 'config' to 'generationConfig' 🚨
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0,
                             "maxOutputTokens": 4096} 
    }
    
    async with session.post(url, headers=headers, json=body, timeout=60) as resp:
        if resp.status != 200:
            text_resp = await resp.text()
            # If a 400 or other error occurs here, it will now show
            raise RuntimeError(f"Gemini API call failed: {resp.status} - {text_resp}")
            
        data = await resp.json()
        
        if data and data.get("candidates"):
             return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
             return json.dumps(data)

# ======================
# 3️⃣ Build prompt
# ======================
def build_prompt(question: str, sections: List[Dict]) -> str:
    sec_text = "\n\n".join(
        [f"PDF: {s['pdf_name']} | Page: {s['page_number']}\n{s['text']}" for s in sections]
    )
    return f"""
You are a healthcare compliance auditor.

Task: Answer the following question using the provided policy sections. Use the sections to find the exact answer.

Question: {question}

Policy Sections:
{sec_text}

Instructions:
- Identify the section(s) that fully answer the question.
- If a section contains a numbered list, bullet points, or criteria, include the full list.
- Include PDF name and page number in evidence.
- Respond ONLY in JSON in this format:
{{
"requirement_met": true/false,
"evidence": "Exact sentences or list items from the sections including PDF name and page number."
}}
"""

# ======================
# 4️⃣ Evaluate a single question
# ======================
async def evaluate_question_async(session, api_key, model, question, top_sections):
    prompt = build_prompt(question, top_sections)
    try:
        resp_text = await call_gemini_async(session, api_key, model, prompt)
        match = re.search(r'\{.*\}', resp_text, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}

        # Prepend PDF name & page number to evidence
        if parsed.get("evidence"):
            evidence_text = parsed["evidence"]
            for sec in top_sections:
                if sec['text'] in evidence_text:
                    evidence_text = evidence_text.replace(sec['text'],
                                                          f"PDF: {sec['pdf_name']} | Page: {sec['page_number']}\n{sec['text']}")
            parsed["evidence"] = evidence_text

        parsed["question"] = question
        return parsed
    except Exception as e:
        return {"question": question, "requirement_met": None, "evidence": f"Error: {e}"}


# ======================
# 5️⃣ Main evaluator
# ======================
async def evaluate_with_semantic_search(api_key: str, questions: List[str], pdf_paths: List[str],
                                        top_k: int = 5, model_name: str = "gemini-2.5-pro") -> List[Dict]:

    # Step 0 — Extract paragraphs from PDFs
    all_sections = []
    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path)
        # Assuming extract_multiple_pdfs_with_pages is adapted for a single file object
        with open(pdf_path, "rb") as f:
            # NOTE: This line requires 'pdf_parser.py' implementation which isn't shown
            pages = extract_multiple_pdfs_with_pages([f]) 
        for page_text, page_num, pdf_name in pages:
            paragraphs = split_into_paragraphs(page_text, page_num, pdf_name)
            all_sections.extend(paragraphs)

    # Step 1 — Load or compute embeddings (Cache check)
    if os.path.exists(EMBEDDING_CACHE_FILE):
        with open(EMBEDDING_CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        for s in all_sections:
            key = f"{s['pdf_name']}|{s['page_number']}|{s['text'][:100]}"
            if key in cached:
                s["embedding"] = cached[key]
    else:
        cached = {}

    # --------------------------------------------------------------------------------------
    # 🚨 CRITICAL FIX: Use a SINGLE aiohttp.ClientSession for ALL API calls 🚨
    # --------------------------------------------------------------------------------------
    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        
        # 1a. Compute new embeddings for PDF chunks
        tasks = []
        for s in all_sections:
            if "embedding" not in s:
                # Uses the active 'session' object
                tasks.append(get_gemini_embedding(session, api_key, s["text"]))
            else:
                # Placeholder for cached embeddings to keep gather() happy
                tasks.append(asyncio.sleep(0, result=s["embedding"])) 
                
        embeddings = await asyncio.gather(*tasks)
        
        # 1b. Update all_sections and the cache
        for s, emb in zip(all_sections, embeddings):
            s["embedding"] = emb
            key = f"{s['pdf_name']}|{s['page_number']}|{s['text'][:100]}"
            cached[key] = emb
    
        # Save cache immediately after all document embeddings are done
        with open(EMBEDDING_CACHE_FILE, "wb") as f:
            pickle.dump(cached, f)

        # 2. Evaluate questions (Now within the single, active session)
        tasks = []
        for q in questions:
            # Embed the question using the active 'session'
            q_emb = await get_gemini_embedding(session, api_key, q)
            
            # Semantic search (retrieval)
            scored_sections = [(cosine_similarity(q_emb, s["embedding"]), s) for s in all_sections]
            scored_sections.sort(reverse=True, key=lambda x: x[0])
            top_sections = [s for _, s in scored_sections[:top_k]]
            
            # Generate final answer using the active 'session'
            tasks.append(evaluate_question_async(session, api_key, model_name, q, top_sections))
            
        results = await asyncio.gather(*tasks)
        
    # The session is automatically closed ONLY after the 'async with' block completes

    return results

# ======================
# 6️⃣ Example usage
# ======================
if __name__ == "__main__":
    import time
    API_KEY = "AIzaSyAUvcpK1WEAEkz6Cf1NeAkg1WqPCpbgANg"  # Replace with your valid key

    questions = [
        "Can a member request a second opinion?",
        "How is the in network specialist physician selected?"
    ]

    pdf_paths = [
        r"C:\Users\disis\LLM_Audit_Model\AA.1000_CEO20250206_v20250201.pdf",
        r"C:\Users\disis\LLM_Audit_Model\GG.1508_CEO20241122_v20241101.pdf"
    ]

    start = time.time()
    results = asyncio.run(evaluate_with_semantic_search(API_KEY, questions, pdf_paths))
    end = time.time()

    print(f"\nEvaluation completed in {end-start:.2f} seconds\n")
    for r in results:
        print("\n--------------------------------")
        print(f"Question: {r['question']}")
        print(f"Requirement Met: {r.get('requirement_met')}")
        print(f"Evidence:\n{r.get('evidence')}\n")