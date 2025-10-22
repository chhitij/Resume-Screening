# Resume Processor

This module provides utilities for processing resumes, analyzing them against job descriptions, and storing them in a vector database for future querying. It uses LangChain components for document loading, text splitting, and embeddings.

## Features
- **Load Resumes**: Supports PDF, DOCX, and TXT formats.
- **Analyze Resumes**: Compares resumes against job descriptions using an LLM (Gemini).
- **Store Resumes**: Saves processed resumes into a Chroma vector store for efficient retrieval.
- **Query Resumes**: Allows querying stored resumes using self-query retrievers.

## Workflow
1. **Load Resumes**:
   - Uses `PyPDFLoader`, `Docx2txtLoader`, or `TextLoader` to load resumes.
2. **Analyze Resumes**:
   - Splits resumes into chunks using `RecursiveCharacterTextSplitter`.
   - Sends each chunk to the Gemini LLM for analysis.
   - Aggregates results into a comprehensive report.
3. **Store Resumes**:
   - Converts text chunks into embeddings using GoogleGenerativeAIEmbeddings.
   - Stores embeddings in a Chroma vector store.
4. **Query Resumes**:
   - Retrieves relevant resumes from the vector store using similarity search.

## Functions
### `load_resume(file_path: str)`
Loads a resume document from the given file path. Supports PDF, DOCX, and TXT formats.

### `analyze_resume(docs, job_description)`
Analyzes resumes against a job description using the Gemini LLM.

### `store_to_vectorstore(text_chunks, persist_directory="chroma_store")`
Stores text chunks into a persistent Chroma vector store.

### `run_self_query(query, persist_directory="chroma_store")`
Retrieves relevant resume chunks from the vector store using a query.

## Example Usage
```python
from resume_processor import load_resume, analyze_resume, store_to_vectorstore

# Load a resume
docs = load_resume("resume.pdf")

# Analyze the resume
job_description = "Looking for a Python developer with AWS experience."
report = analyze_resume(docs, job_description)
print(report)

# Store the resume in a vector store
store_to_vectorstore(docs)
```