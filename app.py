
"""
DocuChat - Chat with your PDF
RAG (Retrieval-Augmented Generation)

Workflow:
1. Extract text from PDF.
2. Split text into overlapping chunks.
3. Generate embeddings and store them in ChromaDB.
4. Retrieve relevant chunks for a question.
5. Send context to Gemini.
6. Generate answers with page citations.
"""

import os
import time
import uuid

import chromadb
import streamlit as st

from google import genai
from google.genai import errors, types
from pypdf import PdfReader


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="DocuChat - AI PDF Assistant",
    page_icon="📄",
    layout="wide"
)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K = 5

MODEL = "gemini-3.8-flash"

# Backup models in case the primary model is temporarily unavailable.
FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite"
]


# =====================================================
# 2. GEMINI API CONFIGURATION
# =====================================================

def get_api_key():

    # First check Streamlit Cloud Secrets
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")

    except Exception:
        api_key = None

    # If not found, check local environment variable
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key is missing. "
            "Configure GEMINI_API_KEY in Streamlit Secrets "
            "or your local environment."
        )

    return api_key


def get_gemini_client():

    api_key = get_api_key()

    return genai.Client(api_key=api_key)


# =====================================================
# 3. PDF TEXT EXTRACTION
# =====================================================

def extract_pages(pdf_file):

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():

            pages.append((page_number, text))

    return pages


# =====================================================
# 4. TEXT CHUNKING
# =====================================================

def chunk_text(
    text,
    size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    start = 0

    while start < len(text):

        chunk = text[start:start + size]

        chunks.append(chunk)

        start += size - overlap

    return chunks


# =====================================================
# 5. VECTOR DATABASE
# =====================================================

def build_index(pdf_file):

    client = chromadb.Client()

    collection = client.create_collection(
        name=f"doc_{uuid.uuid4().hex[:8]}"
    )

    ids = []
    documents = []
    metadata = []

    pages = extract_pages(pdf_file)

    for page_number, text in pages:

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks):

            ids.append(
                f"p{page_number}_c{chunk_number}"
            )

            documents.append(chunk)

            metadata.append({
                "page": page_number
            })

    if not documents:

        return None, 0

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadata
    )

    return collection, len(documents)


# =====================================================
# 6. GEMINI API CALL
# =====================================================

def call_gemini(
    client,
    system_prompt,
    user_prompt
):

    last_error = None

    for model in [MODEL] + FALLBACK_MODELS:

        for attempt in range(3):

            try:

                response = client.models.generate_content(

                    model=model,

                    contents=user_prompt,

                    config=types.GenerateContentConfig(

                        system_instruction=system_prompt,

                        max_output_tokens=1000

                    )

                )

                return response.text

            except errors.ServerError as error:

                last_error = error

                time.sleep(2 ** (attempt + 1))

            except errors.ClientError as error:

                last_error = error

                # Try another model if unavailable or quota exceeded
                if error.code in (404, 429):

                    break

                raise

    raise last_error


# =====================================================
# 7. QUESTION ANSWERING
# =====================================================

def answer_question(question, collection):

    results = collection.query(

        query_texts=[question],

        n_results=min(
            TOP_K,
            collection.count()
        )

    )

    chunks = results["documents"][0]

    metadata = results["metadatas"][0]

    pages = [
        item["page"]
        for item in metadata
    ]

    context = "\n\n".join(

        f"[Page {page}]\n{chunk}"

        for page, chunk in zip(pages, chunks)

    )

    system_prompt = """

    You are DocuChat, an intelligent PDF assistant.

    Answer questions using ONLY the provided document context.

    Rules:

    1. Do not invent information.
    2. If information is unavailable, say so.
    3. Explain answers clearly.
    4. Mention the relevant PDF page numbers.
    5. Keep answers understandable for beginners.

    """

    user_prompt = f"""

    DOCUMENT CONTEXT:

    {context}

    USER QUESTION:

    {question}

    Provide an accurate answer with page references.

    """

    client = get_gemini_client()

    answer = call_gemini(

        client,

        system_prompt,

        user_prompt

    )

    return answer, sorted(set(pages)), chunks


# =====================================================
# 8. APPLICATION INTERFACE
# =====================================================

st.title("📄 DocuChat")

st.subheader(
    "AI-powered PDF Question Answering Assistant"
)

st.write(
    "Upload a PDF and ask questions. "
    "DocuChat retrieves relevant information "
    "and generates answers with page references."
)

st.divider()


# =====================================================
# 9. API KEY CHECK
# =====================================================

try:

    get_api_key()

except ValueError as error:

    st.error(str(error))

    st.info(
        "Configure your Gemini API key in "
        "Streamlit Cloud Secrets or your local environment."
    )

    st.stop()


# =====================================================
# 10. PDF UPLOAD
# =====================================================

uploaded_file = st.file_uploader(

    "📁 Upload your PDF document",

    type=["pdf"]

)


if uploaded_file:

    file_id = (
        uploaded_file.name,
        uploaded_file.size
    )

    if st.session_state.get("file_id") != file_id:

        with st.spinner(
            "Reading and indexing your PDF..."
        ):

            try:

                collection, number_of_chunks = build_index(
                    uploaded_file
                )

                if collection is None:

                    st.error(
                        "No readable text found. "
                        "Scanned PDFs may require OCR."
                    )

                    st.stop()

                st.session_state.collection = collection

                st.session_state.file_id = file_id

                st.session_state.number_of_chunks = (
                    number_of_chunks
                )

                st.session_state.chat_history = []

            except Exception as error:

                st.error(
                    f"PDF processing failed: {error}"
                )

                st.stop()

    st.success(

        f"Successfully indexed "
        f"{st.session_state.number_of_chunks} "
        f"chunks from {uploaded_file.name}"

    )

    st.divider()


    # =================================================
    # 11. CHAT INTERFACE
    # =================================================

    st.subheader("💬 Ask questions about your PDF")

    for message in st.session_state.get(
        "chat_history",
        []
    ):

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    question = st.chat_input(
        "Ask something about your document..."
    )


    if question:

        st.session_state.chat_history.append({

            "role": "user",

            "content": question

        })

        with st.chat_message("user"):

            st.markdown(question)


        with st.chat_message("assistant"):

            with st.spinner("Analyzing your document..."):

                try:

                    answer, pages, chunks = answer_question(

                        question,

                        st.session_state.collection

                    )

                    st.markdown(answer)

                    st.caption(

                        "Sources: Pages "
                        + ", ".join(map(str, pages))

                    )

                    with st.expander(
                        "View retrieved document excerpts"
                    ):

                        for chunk in chunks:

                            st.write(chunk)

                            st.divider()


                    st.session_state.chat_history.append({

                        "role": "assistant",

                        "content": answer

                    })


                except Exception as error:

                    st.error(
                        f"Unable to generate answer: {error}"
                    )


else:

    st.info(
        "👆 Upload a PDF to begin chatting with your document."
    )


# =====================================================
# 12. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("DocuChat")

    st.write(
        "Your personal AI document assistant."
    )

    st.divider()

    st.markdown("### Features")

    st.write("📄 PDF text extraction")

    st.write("🧠 Semantic search")

    st.write("🔍 Vector database")

    st.write("🤖 Gemini AI integration")

    st.write("📚 Page-based citations")

    st.divider()

    if st.button("Clear chat history"):

        st.session_state.chat_history = []

        st.rerun()

    st.caption(
        "Powered by Python, Streamlit, ChromaDB and Gemini."
    )
