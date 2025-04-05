import os
import streamlit as st
from dotenv import load_dotenv  # 👈 Load env vars from .env file
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔑 Load API key from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key not found. Please check your .env file.")
else:
    genai.configure(api_key=api_key)

# 📄 Load and extract text from PDF
def extract_text_from_pdf(pdf_path):
    if not pdf_path.endswith(".pdf"):
        raise ValueError("Invalid file type. Please provide a PDF file.")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    return pages

# 🧠 Create vector store
def create_vector_store(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    return db

# 🤖 Ask question using Gemini AI
def ask_question(query, db):
    docs = db.similarity_search(query)
    relevant_text = "\n".join(x.page_content for x in docs)
    input_prompt = f"Answer the following question based on the given context:\n\nContext:\n{relevant_text}\n\nQuestion: {query}"
    try:
        model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
        response = model.generate_content(input_prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# 🎨 Streamlit UI
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot with Gemini AI")

# 📂 Sidebar for PDF Upload
st.sidebar.header("Upload PDF File")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success("PDF uploaded successfully!")
    pages = extract_text_from_pdf(pdf_path)
    vector_db = create_vector_store(pages)
    
    # 💬 Chat Interface
    query = st.text_input("Ask a question from the PDF:")
    if st.button("Get Answer") and query:
        answer = ask_question(query, vector_db)
        st.write("**Chatbot Answer:**")
        st.write(answer)
