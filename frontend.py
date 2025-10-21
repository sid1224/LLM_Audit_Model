import streamlit as st
import requests
import pandas as pd
import io

# Configuration: Ensure this matches the port used by your running Uvicorn/FastAPI server
FASTAPI_API_URL = "http://localhost:8000/evaluate" 

st.set_page_config(page_title="Compliance Evaluator RAG", layout="wide")

st.title("🛡️ LLM Policy Compliance Evaluator")
st.caption("The UI is built with Streamlit, and the RAG pipeline runs securely on the FastAPI server.")

# --- File Upload Section ---
st.header("1. Upload Documents")

col1, col2 = st.columns(2)

with col1:
    # Multiple files accepted for Policies
    policy_files = st.file_uploader(
        "Upload Policy Documents (PDF, Multiple accepted)", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="policy_uploader"
    )

with col2:
    # Single file accepted for Questions
    questions_file = st.file_uploader(
        "Upload Audit Questions File (PDF, Single file)", 
        type=["pdf"], 
        accept_multiple_files=False,
        key="questions_uploader"
    )

st.divider()

# --- Run Evaluation ---
if st.button("🚀 Run Compliance Audit", use_container_width=True, type="primary"):
    if not policy_files or not questions_file:
        st.error("Please upload both Policy Document(s) and the Audit Questions File.")
    else:
        with st.spinner("Running RAG Pipeline on FastAPI server... This may take a moment."):
            
            # Prepare files for multipart/form-data upload expected by FastAPI
            files = []
            
            # 1. Add Policy Files (key: 'policy_files')
            for file in policy_files:
                # requests expects (file_name, file_data_bytes, mime_type)
                files.append(('policy_files', (file.name, file.getvalue(), 'application/pdf')))
            
            # 2. Add Questions File (key: 'questions_file')
            files.append(('questions_file', (questions_file.name, questions_file.getvalue(), 'application/pdf')))

            try:
                # Send the POST request to the running FastAPI backend
                response = requests.post(FASTAPI_API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        st.session_state['results'] = results
                        st.success(f"Audit Complete! {len(results)} requirements evaluated.")
                    else:
                        st.warning("Audit completed, but no results were returned by the backend.")

                else:
                    # Handle errors returned from the FastAPI server (e.g., HTTPException details)
                    error_detail = response.json().get('detail', response.text)
                    st.error(f"FastAPI Server Error ({response.status_code}): {error_detail}")

            except requests.exceptions.ConnectionError:
                st.error(f"Connection Error: Could not connect to the FastAPI server at {FASTAPI_API_URL}. Ensure the backend is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


# --- Display Results ---
if 'results' in st.session_state and st.session_state['results']:
    st.header("2. Audit Results")

    # Convert results to DataFrame for better display
    df = pd.DataFrame(st.session_state['results'])
    
    # Clean up and select columns
    df = df[['question', 'requirement_met', 'evidence']]
    df.columns = ['Requirement Question', 'Met?', 'Verbatim Evidence']

    # Custom styling for the 'Met?' column: Green for True, Red for False
    def color_met(val):
        color = 'green' if val == 'True' else 'red' if val == 'False' else 'gray'
        return f'background-color: {color}; color: white; font-weight: bold; padding: 4px 8px; border-radius: 4px;'
    
    st.dataframe(
        df.style.apply(lambda x: [color_met(x['Met?'])] * len(x) if x.name == 'Met?' else [''] * len(x), axis=1), 
        use_container_width=True,
        height=500
    )
