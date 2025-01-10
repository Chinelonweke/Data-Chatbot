import fitz  # PyMuPDF for PDF processing
from docx import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


# A function to load a file and extract text based on file type
def load_file(file_path):
    """
    Load a file (PDF or DOCX) and return its text content.
    """
    file_type = file_path.split('.')[-1].lower()
    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type == "docx":
        return extract_text_from_docx(file_path)
    else:
        return "Unsupported file type"


# Function to process the user query against the FAISS index
def process_user_query(vector_store, user_query):
    """
    Process the user query using a prebuilt FAISS index and return the most relevant answer.
    """
    # Perform a similarity search for the user query in the FAISS vector store
    results = vector_store.similarity_search(user_query, k=3)  # Adjust 'k' based on desired results
    if results:
        # Combine the top results into one response
        response = "\n".join([result.page_content for result in results])
        return response
    else:
        return "No relevant information found."
