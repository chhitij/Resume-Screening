"""
resume_processor_perplexity.py

Edited version of your resume processing module to use Perplexity for LLM (chat/completions)
while keeping Google embeddings for vector storage. This file:
- Provides a `perplexity_chat` helper with retry/backoff
- Replaces Gemini `llm.invoke()` calls with `perplexity_chat()` usage in `analyze_resume`
- Keeps `store_to_vectorstore` using GoogleGenerativeAIEmbeddings
- Simplifies `run_self_query` to use Chroma similarity search and (optionally) Perplexity
  for lightweight reranking/interpretation.

Requirements:
- PERPLEXITY_API_KEY in your .env
- Keep your GOOGLE_API_KEY in .env (for embeddings)
- requests

Usage:
- import functions from this file and call `load_resume`, `analyze_resume(docs, job_description)`,
  `store_to_vectorstore(chunks)` and `run_self_query(query)` as needed.

"""

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv
import requests
import time
from typing import List

load_dotenv()

PERPLEXITY_MODEL = "sonar-pro"  # or other available Perplexity models

# ----------------- Keys -----------------
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")  # ensure env var is set

print("PERPLEXITY_API_KEY:", PERPLEXITY_API_KEY)

# ----------------- Embeddings (Google) -----------------
# Keep using Google embeddings (stable for vectorization)
embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ----------------- Perplexity helper -----------------
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

def perplexity_chat(messages: List[dict],
                    model: str = PERPLEXITY_MODEL,
                    max_tokens: int = 800,
                    temperature: float = 0.2,
                    retries: int = 3,
                    backoff_seconds: int = 5):
    """Send chat messages to Perplexity and return assistant text.

    Retries on transient errors and basic rate limit handling.
    """
    if not PERPLEXITY_API_KEY:
        raise RuntimeError("PERPLEXITY_API_KEY is not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # Defensive parsing for common response shape
            choice = data.get("choices", [{}])[0]
            message = choice.get("message") or choice.get("text") or {}
            if isinstance(message, dict):
                return message.get("content", "")
            return str(message)
        except requests.HTTPError as http_err:
            status = getattr(http_err.response, "status_code", None)
            # Handle simple rate-limit status codes
            if status in (429, 503):
                wait = backoff_seconds * attempt
                print(f"Perplexity rate limit or service error ({status}). Backing off {wait}s (attempt {attempt}).")
                time.sleep(wait)
                continue
            # For other HTTP errors, raise
            raise
        except requests.RequestException as e:
            wait = backoff_seconds * attempt
            print(f"Perplexity request error: {e}. Backing off {wait}s (attempt {attempt}).")
            time.sleep(wait)
            continue
    raise RuntimeError("Perplexity API request failed after retries.")

# ----------------- File loaders -----------------

def load_resume(file_path: str):
    """Load a resume document from disk. Returns a list of LangChain Document objects."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file format.")
    return loader.load()

# ----------------- Analyze resume (uses Perplexity) -----------------

def analyze_resume(docs: List[object], job_description: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """Analyze resumes against a job description using Perplexity LLM.

    Args:
        docs: list of LangChain Document objects (e.g., from load_resume)
        job_description: job description string
    Returns:
        str: aggregated analysis
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    full_analysis = ""
    for i, chunk in enumerate(chunks):
        prompt = f"""
Compare this resume chunk with the job description. Provide exactly the following sections:\n
1. Suitability Score (out of 100)\n2. Skills Matched\n3. Experience Relevance\n4. Education Evaluation\n5. Strengths\n6. Weaknesses\n7. Final Recommendation\n
Job Description:\n{job_description}\n\nResume chunk:\n{chunk.page_content}\n"""
        messages = [
            {"role": "system", "content": "You are an expert technical recruiter. Score objectively and be concise."},
            {"role": "user", "content": prompt}
        ]
        try:
            result_text = perplexity_chat(messages, model=PERPLEXITY_MODEL, max_tokens=800)
        except Exception as e:
            # If Perplexity fails, record the error and continue
            result_text = f"[ERROR calling Perplexity: {e}]"
        full_analysis += f"--- Chunk {i+1} ---\n" + result_text + "\n\n"

    return full_analysis

# ----------------- Vector store functions (unchanged except minor safety) -----------------

def store_to_vectorstore(text_chunks: List[object], persist_directory: str = "chroma_store"):
    """Save text chunks into a persistent Chroma vector store using Google embeddings."""
    texts = [chunk.page_content for chunk in text_chunks]
    metadatas = [{"source": f"resume_chunk_{i}"} for i in range(len(texts))]

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

# ----------------- Simple self-query / retrieval -----------------

def run_self_query(query: str, persist_directory: str = "chroma_store", k: int = 5, rerank_with_perplexity: bool = False):
    """Retrieve relevant resume chunks from Chroma using similarity search.

    If rerank_with_perplexity=True, the top-k results will be sent to Perplexity
    with the original query for a light-weight ordering/selection.
    """
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    # similarity_search returns a list of LangChain Documents
    results = vectorstore.similarity_search(query, k=k)

    if not rerank_with_perplexity:
        return results

    # Build a prompt to rerank the candidate chunks
    combined = "\n\n---\n\n".join([f"Chunk {i+1}:\n{d.page_content}" for i, d in enumerate(results)])
    prompt = f"You are an expert recruiter. Given the user query:\n{query}\n\nRerank the following resume chunks in order of relevance (most relevant first). Return a JSON array of indices (1-based) in the new order and a one-sentence reason for the top 1 chunk.\n\n{combined}"
    messages = [
        {"role": "system", "content": "You are an expert recruiter. Be precise and return valid JSON."},
        {"role": "user", "content": prompt}
    ]
    try:
        rerank_text = perplexity_chat(messages, model=PERPLEXITY_MODEL, max_tokens=400)
        # try to extract JSON from the response
        import json
        # naive extraction: find first '[' and last ']'
        start = rerank_text.find('[')
        end = rerank_text.rfind(']')
        if start != -1 and end != -1 and end > start:
            order = json.loads(rerank_text[start:end+1])
            # reorder results according to 'order'
            reordered = [results[idx-1] for idx in order if 1 <= idx <= len(results)]
            return reordered
    except Exception as e:
        print(f"Rerank with Perplexity failed: {e}")

    # fallback to original results
    return results

# ===================== END OF FILE =====================
