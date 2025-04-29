# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from huggingface_hub import InferenceClient
from langchain_core.language_models import LLM
import os

# ----------------------------- Custom LangChain Wrapper -----------------------------

class HFInferenceLLM(LLM):
    def __init__(self, api_key, model):
        self.client = InferenceClient(provider="nebius", api_key=api_key)
        self.model = model

    def _call(self, prompt: str, **kwargs) -> str:
        response = self.client.text_generation(
            model=self.model,
            prompt=prompt,
            max_new_tokens=256,
            temperature=0.7
        )
        return response

    @property
    def _llm_type(self) -> str:
        return "huggingface-inference-client"

# ----------------------------- App Logic -----------------------------

def load_document(uploaded_file):
    file_path = f"temp_docs/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        st.error("Unsupported file format.")
        return None

    return loader.load()

def process_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever()

def answer_question(user_query, retriever, api_key):
    llm = HFInferenceLLM(api_key=api_key, model="microsoft/Phi-3.5-mini-instruct")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.run(user_query)
    return result

# ----------------------------- Streamlit UI -----------------------------

st.set_page_config(page_title="RAG Q&A App", layout="wide")
st.title("🧠 LLM RAG Q&A with Phi-3.5-mini (Nebius HF)")

hf_api_key = st.text_input("Enter your Hugging Face API Key (Nebius)", type="password")

uploaded_file = st.file_uploader("Upload a Document (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

if uploaded_file and hf_api_key:
    docs = load_document(uploaded_file)
    if docs:
        st.success("Document Loaded Successfully.")
        retriever = process_docs(docs)

        query = st.text_input("Ask a question about the document:")
        if st.button("Submit") and query:
            with st.spinner("Generating answer..."):
                response = answer_question(query, retriever, api_key=hf_api_key)
                st.markdown("### 📌 Answer")
                st.write(response)
else:
    st.warning("Please upload a document and enter your HF API key.")
