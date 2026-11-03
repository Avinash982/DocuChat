
import os
import time
import uuid
import hashlib
from io import BytesIO

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
    page_icon="📚",
    layout="wide"
)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K = 5

MODEL = "gemini-3.8-flash"

FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite"
]


# =====================================================
# 2. API KEY CONFIGURATION
# =====================================================

def get_api_key():

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

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

    return genai.Client(api_key=get_api_key())


# =====================================================
# 3. PDF EXTRACTION
# =====================================================

def extract_pages(pdf_bytes):

    reader = PdfReader(BytesIO(pdf_bytes))

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
# 5. MULTIPLE PDF INDEXING
# =====================================================

def build_index(uploaded_files):

    client = chromadb.Client()

    collection = client.create_collection(
        name=f"doc_{uuid.uuid4().hex[:12]}"
    )

    ids = []
    documents = []
    metadata = []

    total_pages = 0

    for uploaded_file in uploaded_files:

        pdf_bytes = uploaded_file.getvalue()

        pages = extract_pages(pdf_bytes)

        total_pages += len(pages)

        for page_number, text in pages:

            chunks = chunk_text(text)

            for chunk_number, chunk in enumerate(chunks):

                ids.append(uuid.uuid4().hex)

                documents.append(chunk)

                metadata.append({
                    "source": uploaded_file.name,
                    "page": page_number,
                    "chunk": chunk_number
                })

    if not documents:

        return None, 0, 0

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadata
    )

    return collection, len(documents), total_pages


# =====================================================
# 6. GEMINI API
# =====================================================

def call_gemini(client, system_prompt, user_prompt):

    last_error = None

    for model in [MODEL] + FALLBACK_MODELS:

        try:

            response = client.models.generate_content(

                model=model,

                contents=user_prompt,

                config=types.GenerateContentConfig(

                    system_instruction=system_prompt,

                    max_output_tokens=1500

                )

            )

            if response.text:
                return response.text

            raise RuntimeError("Gemini returned an empty response.")

        except errors.ClientError as error:

            last_error = error

            if error.code in (404, 429):

                continue

            raise

        except errors.ServerError as error:

            last_error = error

            time.sleep(2)

    if last_error:
        raise last_error

    raise RuntimeError("No Gemini model was available.")


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

    context_parts = []

    sources = []

    for chunk, item in zip(chunks, metadata):

        filename = item["source"]

        page = item["page"]

        context_parts.append(
            f"[Document: {filename} | Page: {page}]\n{chunk}"
        )

        sources.append((filename, page))

    context = "\n\n".join(context_parts)

    system_prompt = """
    You are DocuChat, an intelligent PDF assistant.

    Answer questions using ONLY the supplied document context.

    Rules:

    1. Never invent information.
    2. If the answer is unavailable, say so.
    3. Use information from all relevant documents.
    4. Compare documents when requested.
    5. Mention document filenames and page numbers.
    6. Explain answers clearly.
    7. Keep answers understandable for beginners.
    """

    user_prompt = f"""
    DOCUMENT CONTEXT:

    {context}

    USER QUESTION:

    {question}

    Answer accurately using the supplied documents.
    Include relevant document names and page references.
    """

    client = get_gemini_client()

    answer = call_gemini(
        client,
        system_prompt,
        user_prompt
    )

    return answer, sorted(set(sources)), chunks


# =====================================================
# 8. APPLICATION HEADER
# =====================================================

st.title("📚 DocuChat")

st.subheader(
    "AI-Powered Multi-PDF Question Answering Assistant"
)

st.write(
    "Upload multiple PDF documents and ask questions "
    "across all your documents using Gemini AI."
)

st.divider()


# =====================================================
# 9. API KEY CHECK
# =====================================================

try:

    get_api_key()

except ValueError as error:

    st.error(str(error))

    st.stop()


# =====================================================
# 10. MULTIPLE PDF UPLOAD
# =====================================================

uploaded_files = st.file_uploader(

    "📁 Upload one or more PDF documents",

    type=["pdf"],

    accept_multiple_files=True

)


if uploaded_files:

    # Create a unique identifier for the uploaded collection.

    file_signature = hashlib.sha256()

    for file in uploaded_files:

        file_signature.update(file.name.encode())

        file_signature.update(file.getvalue())

    current_signature = file_signature.hexdigest()


    # Rebuild the index only when uploaded PDFs change.

    if st.session_state.get("file_signature") != current_signature:

        with st.spinner(
            "Reading and indexing all your PDF documents..."
        ):

            try:

                collection, number_of_chunks, total_pages = (
                    build_index(uploaded_files)
                )

                if collection is None:

                    st.error(
                        "No readable text found in the uploaded PDFs. "
                        "Scanned PDFs may require OCR."
                    )

                    st.stop()

                st.session_state.collection = collection

                st.session_state.file_signature = current_signature

                st.session_state.number_of_chunks = number_of_chunks

                st.session_state.total_pages = total_pages

                st.session_state.chat_history = []

                st.session_state.uploaded_names = [
                    file.name for file in uploaded_files
                ]

            except Exception as error:

                st.error(
                    f"PDF processing failed: {error}"
                )

                st.stop()


    # =================================================
    # 11. DOCUMENT INFORMATION
    # =================================================

    st.success(
        f"Successfully indexed {len(uploaded_files)} PDF documents!"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "PDF Documents",
        len(uploaded_files)
    )

    col2.metric(
        "Total Pages",
        st.session_state.total_pages
    )

    col3.metric(
        "Text Chunks",
        st.session_state.number_of_chunks
    )

    with st.expander("📂 View uploaded documents"):

        for filename in st.session_state.uploaded_names:

            st.write(f"📄 {filename}")


    st.divider()


    # =================================================
    # 12. CHAT INTERFACE
    # =================================================

    st.subheader("💬 Chat with your documents")

    for message in st.session_state.get(
        "chat_history", []
    ):

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    question = st.chat_input(
        "Ask a question about your PDFs..."
    )


    if question:

        st.session_state.chat_history.append({

            "role": "user",

            "content": question

        })

        with st.chat_message("user"):

            st.markdown(question)


        with st.chat_message("assistant"):

            with st.spinner(
                "Searching documents and generating your answer..."
            ):

                try:

                    answer, sources, chunks = answer_question(

                        question,

                        st.session_state.collection

                    )

                    st.markdown(answer)

                    st.markdown("### 📚 Sources")

                    for filename, page in sources:

                        st.write(
                            f"📄 **{filename}** — Page {page}"
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
        "👆 Upload one or more PDFs to start chatting."
    )


# =====================================================
# 13. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("📚 DocuChat")

    st.write(
        "Your personal AI document assistant."
    )

    st.divider()

    st.markdown("### Features")

    st.write("📄 Multiple PDF uploads")

    st.write("🧠 Semantic document search")

    st.write("🔍 ChromaDB vector database")

    st.write("🤖 Gemini AI integration")

    st.write("📚 Document and page citations")

    st.write("🔄 Cross-document question answering")

    st.divider()

    if st.button("Clear chat history"):

        st.session_state.chat_history = []

        st.rerun()

    st.caption(
        "Powered by Python, Streamlit, ChromaDB and Gemini."
    )