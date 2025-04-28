import streamlit as st
import os
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

# Load API key securely
Rag_QA = st.secrets["RAG_QA_KEY"]

# Set the Hugging Face API token as an environment variable
os.environ['HUGGINGFACEHUB_API_TOKEN'] = Rag_QA

def process_input(input_type, input_data):
    """Processes different input types and returns a vectorstore."""
    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        documents = loader.load()
    elif input_type == "PDF":
        pdf_reader = PdfReader(input_data)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        documents = text
    elif input_type == "Text":
        documents = input_data
    elif input_type == "DOCX":
        doc = Document(input_data)
        text = "\n".join([para.text for para in doc.paragraphs])
        documents = text
    elif input_type == "TXT":
        text = input_data.read().decode('utf-8')
        documents = text
    else:
        raise ValueError("Unsupported input type")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    if input_type == "Link":
        texts = text_splitter.split_documents(documents)
        texts = [str(doc.page_content) for doc in texts]
    else:
        texts = text_splitter.split_text(documents)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)

    vector_store = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_texts(texts)
    return vector_store

def answer_question(vectorstore, query):
    """Answers a question based on the provided vectorstore."""
    llm = HuggingFaceEndpoint(
        repo_id='microsoft/Phi-3.5-mini-instruct',
        token=Rag_QA,
        temperature=0.6,
        task="text-generation"
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    answer = qa({"query": query})
    return answer["result"]

def main():
    st.title("RAG Q&A App")

    input_type = st.selectbox("Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])
    if input_type == "Link":
        url = st.text_input("Enter the URL")
        input_data = url
    elif input_type == "Text":
        input_data = st.text_area("Enter the text")
    else:
        input_data = st.file_uploader(f"Upload a {input_type} file", type=[input_type.lower()])

    if st.button("Process Document"):
        if input_data is not None:
            vectorstore = process_input(input_type, input_data)
            st.session_state["vectorstore"] = vectorstore
        else:
            st.warning("Please upload or enter the data.")

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question")
        if st.button("Submit Question"):
            answer = answer_question(st.session_state["vectorstore"], query)
            st.success(answer)

if __name__ == "__main__":
    main()
