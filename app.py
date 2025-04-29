# Set environment variables at the top before importing Streamlit
import os
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import streamlit as st
import faiss
import numpy as np
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

# Load API key securely from Streamlit Secrets
Rag_QA = st.secrets["Rag_QA"]

def process_input(input_type, input_data):
    """Processes different input types and returns a vectorstore."""
    
    # Load documents based on input type
    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        documents = loader.load()
        documents = [doc.page_content for doc in documents]
    elif input_type == "PDF":
        pdf_reader = PdfReader(input_data)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
        documents = [text]
    elif input_type == "Text":
        documents = [input_data]
    elif input_type == "DOCX":
        doc = Document(input_data)
        text = "\n".join(para.text for para in doc.paragraphs)
        documents = [text]
    elif input_type == "TXT":
        text = input_data.read().decode('utf-8')
        documents = [text]
    else:
        raise ValueError("Unsupported input type.")

    # Split documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = []
    for doc in documents:
        texts.extend(text_splitter.split_text(doc))
    
    # Load the HuggingFace embedding model
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    # Initialize FAISS index
    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)

    # Create FAISS vectorstore
    vector_store = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    # Add texts to vectorstore
    vector_store.add_texts(texts)
    return vector_store

def answer_question(vectorstore, query):
    """Answers a question based on the provided vectorstore."""
    
    # Initialize the LLM
    llm = HuggingFaceEndpoint(
        repo_id='microsoft/Phi-3.5-mini-instruct',
        token=Rag_QA,
        temperature=0.6,
        task="text-generation"
    )
    
    # Set up retrieval QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    
    # Query the QA chain
    answer = qa({"query": query})
    return answer["result"]

def main():
    st.title("RAG Q&A Application")

    # User input for document source
    input_type = st.selectbox("Select Input Type:", ["Link", "PDF", "Text", "DOCX", "TXT"])
    
    input_data = None
    if input_type == "Link":
        input_data = st.text_input("Enter the URL:")
    elif input_type == "Text":
        input_data = st.text_area("Enter the Text:")
    else:
        input_data = st.file_uploader(f"Upload your {input_type} file:", type=[input_type.lower()])

    # Process document
    if st.button(" Process Document"):
        if input_data:
            with st.spinner("Processing document and creating knowledge base..."):
                vectorstore = process_input(input_type, input_data)
                st.session_state["vectorstore"] = vectorstore
            st.success("Document processed successfully! You can now ask questions.")
        else:
            st.warning("⚠️ Please upload a file or provide input data.")

    # Ask questions
    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question:")
        if st.button(" Submit Question"):
            if query.strip():
                with st.spinner("Finding the best answer..."):
                    answer = answer_question(st.session_state["vectorstore"], query)
                st.success(answer)
            else:
                st.warning("⚠️ Please enter a question to submit.")

if __name__ == "__main__":
    main()
