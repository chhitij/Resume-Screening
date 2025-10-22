# Loaders & splitters 
# PyPDFLoader: Loads PDF files and extracts text content.
# Docx2txtLoader: Loads DOCX files and extracts text content.
# TextLoader: Loads plain text files with specified encoding.
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
# RecursiveCharacterTextSplitter: Splits long documents into smaller chunks for processing.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# OS and dotenv for environment variables
# os: Provides functions to interact with the operating system.
# dotenv: Loads environment variables from a .env file.
import os
from dotenv import load_dotenv

# Vector store
# Chroma: A vector store for storing and retrieving embeddings.
from langchain_community.vectorstores import Chroma  

# Self-query & schema tools 
# AttributeInfo: Describes metadata fields for self-querying.
# SelfQueryRetriever: Retrieves relevant documents based on a query and metadata.
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# ====== GEMINI: Chat + Embeddings ======
# ChatGoogleGenerativeAI: A chat-based LLM for generating responses.
# GoogleGenerativeAIEmbeddings: Generates embeddings using Google's Text Embedding model.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ----------------- Keys -----------------
# Load environment variables from a .env file.
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # ensure env var is set


# ----------------- Models ----------------
# Embeddings: Google Text Embedding 004
embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Chat LLM: Gemini 1.5 Flash (fast & capable). You can switch to "gemini-1.5-pro".
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# ----------------- Functions -------------
# Load resumes in different formats (unchanged)
def load_resume(file_path):
    """
    Load a resume document from the given file path.

    Args:
        file_path (str): Path to the resume file.

    Returns:
        Document: The loaded resume document.

    Supported formats:
    - PDF: Uses PyPDFLoader to extract text from PDF files.
    - DOCX: Uses Docx2txtLoader to extract text from Word documents.
    - TXT: Uses TextLoader to load plain text files.
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        import docx2txt  # ensure installed
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file format.")
    return loader.load()

# Analyze the resume using Gemini (minimal change: llm now Gemini)
def analyze_resume(docs, job_description):
    """
    Analyze resumes against a job description using the Gemini LLM.

    This function processes a list of resume documents by splitting them into smaller
    chunks, sending each chunk to the Gemini language model for analysis, and
    aggregating the results into a comprehensive analysis report.

    Args:
        docs (List[Document]): A list of resume documents to analyze. Each document
            should be a LangChain `Document` object containing the resume content.
        job_description (str): The job description to compare the resumes against.

    Returns:
        str: A string containing the full analysis report. The report includes:
            - Suitability Score (out of 100)
            - Skills Matched
            - Experience Relevance
            - Education Evaluation
            - Strengths
            - Weaknesses
            - Final Recommendation

    Process:
    1. Splits the input documents into smaller chunks using the
       `RecursiveCharacterTextSplitter` to ensure the content fits within the
       token limits of the language model.
    2. Constructs a prompt for each chunk, including the job description and
       the chunk content.
    3. Sends the prompt to the Gemini LLM using the `invoke` method.
    4. Aggregates the responses from the LLM into a single analysis report.

    Example:
        >>> from langchain.schema import Document
        >>> docs = [Document(page_content="Sample resume content")]
        >>> job_description = "Looking for a software engineer with Python skills."
        >>> report = analyze_resume(docs, job_description)
        >>> print(report)
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    full_analysis = ""
    for chunk in chunks:
        prompt = f"""
Compare this resume with the job description. Give:
1. Suitability Score (out of 100)
2. Skills Matched
3. Experience Relevance
4. Education Evaluation
5. Strengths
6. Weaknesses
7. Final Recommendation

Job Description:
{job_description}

Resume:
{chunk.page_content}
"""
        result = llm.invoke(prompt)  # Gemini call
        full_analysis += result.content + "\n\n"
    return full_analysis

# Store text chunks into ChromaDB (embeddings now Google)
def store_to_vectorstore(text_chunks, persist_directory="chroma_store"):
    """
    Save text chunks into a persistent Chroma vector store.

    This function takes a list of text chunks, converts them into embeddings
    using GoogleGenerativeAIEmbeddings, and stores them in a Chroma vector
    database. The database is then saved to disk for future use.

    Args:
        text_chunks (List[Document]): A list of text chunks to store. Each chunk
            should be a LangChain `Document` object containing the text content.
        persist_directory (str): The directory where the Chroma database will
            be saved.

    Returns:
        Chroma: The persistent Chroma vector store instance.

    Process:
    1. Extracts the text content from each chunk.
    2. Converts the text into embeddings using the GoogleGenerativeAIEmbeddings model.
    3. Stores the embeddings in a Chroma vector store along with metadata.
    4. Saves the vector store to the specified directory.

    Example:
        >>> from langchain.schema import Document
        >>> chunks = [Document(page_content="Sample text chunk")]
        >>> vectordb = store_to_vectorstore(chunks, persist_directory="./chroma_db")
        >>> print("Vector store saved at:", vectordb.persist_directory)
    """
    texts = [chunk.page_content for chunk in text_chunks]
    metadatas = [{"source": f"resume_chunk_{i}"} for i in range(len(texts))]

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embedding,                 # Google embeddings
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

# Use SelfQueryRetriever to interpret and fetch relevant chunks (llm now Gemini)
def run_self_query(query, persist_directory="chroma_store"):
    """
    Retrieve relevant resume chunks from the vector store using a query.

    This function loads a Chroma vector store from disk, interprets the query
    using the Gemini LLM, and fetches the most relevant resume chunks based
    on the query.

    Args:
        query (str): The query to interpret and execute. For example, "Find
            resumes with Python experience."
        persist_directory (str): The directory where the Chroma database is
            stored.

    Returns:
        List[Document]: A list of relevant resume chunks matching the query.

    Process:
    1. Loads the Chroma vector store from the specified directory.
    2. Uses the Gemini LLM to interpret the query and match it against the
       stored embeddings.
    3. Fetches and returns the most relevant resume chunks.

    Example:
        >>> query = "Find resumes with Python experience."
        >>> results = run_self_query(query, persist_directory="./chroma_db")
        >>> for doc in results:
        >>>     print(doc.page_content)
    """
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding         # Google embeddings
    )

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="Where the chunk is from",
            type="string"
        )
    ]

    document_content_description = "This represents a chunk of a resume."

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        search_type="mmr"
    )

    return retriever.get_relevant_documents(query)

# ===================== END OF resume_processor.py =====================
