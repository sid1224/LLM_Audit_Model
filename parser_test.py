# test_extraction.py
from pdf_parser import extract_pdf_text

import os

# print("Current working directory:", os.getcwd())
# print("Files here:", os.listdir())

# Path to your PDF (replace with your file name)
file_path = r"C:\Users\disis\LLM_Audit_Model\AA.1000_CEO20250206_v20250201.pdf"

# Open and extract text
with open(file_path, "rb") as f:
    text = extract_pdf_text(f)

# Check how much text was extracted
print("✅ Text extraction complete.")
print(f"Total characters extracted: {len(text)}")

# Preview the first 1000 characters to verify it's readable
print("\n--- Sample Output ---\n")
print(text[:2000])
