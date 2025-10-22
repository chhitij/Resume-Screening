# Resume Processor with Perplexity

This module extends the resume processing functionality by integrating Perplexity as the LLM for analyzing resumes. It retains Google embeddings for vector storage and adds retry/backoff mechanisms for Perplexity API calls.

## Features
- **Load Resumes**: Supports PDF, DOCX, and TXT formats.
- **Analyze Resumes**: Compares resumes against job descriptions using Perplexity LLM.
- **Store Resumes**: Saves processed resumes into a Chroma vector store for efficient retrieval.
- **Query Resumes**: Allows querying stored resumes with optional Perplexity-based reranking.

## Workflow
1. **Load Resumes**:
   - Uses `PyPDFLoader`, `Docx2txtLoader`, or `TextLoader` to load resumes.
2. **Analyze Resumes**:
   - Splits resumes into chunks using `RecursiveCharacterTextSplitter`.
   - Sends each chunk to the Perplexity LLM for analysis.
   - Aggregates results into a comprehensive report.
3. **Store Resumes**:
   - Converts text chunks into embeddings using GoogleGenerativeAIEmbeddings.
   - Stores embeddings in a Chroma vector store.
4. **Query Resumes**:
   - Retrieves relevant resumes from the vector store using similarity search.
   - Optionally reranks results using Perplexity.

## Functions
### `load_resume(file_path: str)`
Loads a resume document from the given file path. Supports PDF, DOCX, and TXT formats.

### `analyze_resume(docs, job_description)`
Analyzes resumes against a job description using the Perplexity LLM.

### `store_to_vectorstore(text_chunks, persist_directory="chroma_store")`
Stores text chunks into a persistent Chroma vector store.

### `run_self_query(query, persist_directory="chroma_store", rerank_with_perplexity=False)`
Retrieves relevant resume chunks from the vector store using a query. Optionally reranks results using Perplexity.

## Example Usage
```python
from resume_processor_perplexity import load_resume, analyze_resume, store_to_vectorstore

# Load a resume
docs = load_resume("resume.pdf")

# Analyze the resume
job_description = "Looking for a Python developer with AWS experience."
report = analyze_resume(docs, job_description)
print(report)

# Store the resume in a vector store
store_to_vectorstore(docs)
```

## Design Flow
Below is a visual representation of the workflow:

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/8fccf487-1a7c-4630-9bda-3cbca2f88931" />

1. **Load Resumes**: Extracts text from resumes.
2. **Analyze Resumes**: Sends chunks to Perplexity for scoring and analysis.
3. **Store Resumes**: Embeds and stores resumes in a vector database.
4. **Query Resumes**: Retrieves and optionally reranks results.

---
Ensure the `PERPLEXITY_API_KEY` is set in your environment variables for the Perplexity API to function.
