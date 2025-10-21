# Compliance Evaluator (Local Setup)

This repository provides a **Gemini-based Compliance Evaluator** for healthcare or insurance policy documents. It includes:

- PDF parsing and extraction.
- Paragraph-level semantic search with Gemini embeddings.
- Compliance evaluation with an LLM.
- FastAPI backend for serving API endpoints.
- Frontend integration (Streamlit or local script) to view results.

---

## Table of Contents

1. [Clone the Repository](#clone-the-repository)  
2. [Set up Python Environment](#set-up-python-environment)  
3. [Install Dependencies](#install-dependencies)  
4. [Set Environment Variables](#set-environment-variables)  
5. [Run PDF Parsing Scripts](#run-pdf-parsing-scripts)  
6. [Run LLM Compliance Evaluator](#run-llm-compliance-evaluator)  
7. [Run Backend API](#run-backend-api)  
8. [Run Frontend](#run-frontend)  
9. [Upload Files and Get Results](#upload-files-and-get-results)  
10. [Caching and Tips](#caching-and-tips)  

---

## Clone the Repository

# Create a virtual environment
python -m venv ds_env

# Activate the environment
# Windows
ds_env\Scripts\activate
# Mac/Linux
source ds_env/bin/activate

# Install the required files
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
# Windows PowerShell
$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"

# Mac/Linux
export GEMINI_API_KEY="YOUR_API_KEY_HERE"

```bash

