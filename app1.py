import streamlit as st
import hashlib
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document  # For handling Word documents
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


# Hashing Function for Passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Authentication Functions
def register_user(username, password):
    # Initialize user_data when registering a new user
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}
        
    hashed_password = hash_password(password)
    st.session_state.user_data[username] = hashed_password
    st.success("Registration successful! Please log in.")


def authenticate_user(username, password):
    # Check if user_data is initialized
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}

    if not username or not password:
        st.error("Username and password cannot be empty.")
        return False

    hashed_password = hash_password(password)
    if username in st.session_state.user_data and st.session_state.user_data[username] == hashed_password:
        return True
    else:
        st.error("Invalid username or password.")
        return False


# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to extract text from Word documents (.docx)
def get_word_text(word_docs):
    text = ""
    for doc in word_docs:
        doc = Document(doc)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text


# Function to extract text from text files (.txt)
def get_text_file_text(text_files):
    text = ""
    for txt in text_files:
        text += txt.read().decode("utf-8") + "\n"
    return text


# Function to split text into chunks for processing
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function to generate a vectorstore from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Function to get a conversational chain for the chat model
def get_conversation_chain(vectorstore):
    llm = ChatGroq()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Process the conversation and display chat history
def handle_userinput(user_question):
    # Get the response from the conversation chain
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    # Loop through the chat history and display messages
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User's input
            st.write(f"**🤖❓:** {message.content}")
        else:  # Model's response
            # If the response starts with "Based on the information", remove it and start with a direct answer
            if message.content.lower().startswith("based on the information"):
                # Remove the unnecessary phrase and directly address the user's query
                answer = message.content[message.content.find('.')+1:].strip()
                st.write(f"**🤖:** {answer}")
            else:
                # If no unnecessary phrase is present, just show the response
                st.write(f"**🤖:** {message.content}")



def main():
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    load_dotenv()
    st.set_page_config(page_title="Chat with Documents", page_icon=":books:")

    # Sidebar Login/Register
    with st.sidebar:
        st.subheader("Login/Register")
        choice = st.radio("Select an option", ["Login", "Register"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if choice == "Register":
            if st.button("Register"):
                if username and password:
                    register_user(username, password)
                else:
                    st.warning("Please provide both username and password.")
        elif choice == "Login":
            if st.button("Login"):
                if username and password:
                    if authenticate_user(username, password):
                        st.success("Login successful!")
                        st.session_state.logged_in_user = username
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.warning("Please provide both username and password.")

    # Main app after login
    if "logged_in_user" in st.session_state:
        st.header(f"Welcome, {st.session_state.logged_in_user}!")

        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Your Documents")
            uploaded_files = st.file_uploader(
                "Upload your PDFs, Word docs, and text files here", accept_multiple_files=True
            )
            if st.button("Process"):
                with st.spinner("Processing..."):
                    raw_text = ""
                    # Separate files based on file type
                    pdf_files = [file for file in uploaded_files if file.type == "application/pdf"]
                    word_files = [file for file in uploaded_files if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
                    text_files = [file for file in uploaded_files if file.type == "text/plain"]

                    # Extract text from files
                    if pdf_files:
                        raw_text += get_pdf_text(pdf_files)
                    if word_files:
                        raw_text += get_word_text(word_files)
                    if text_files:
                        raw_text += get_text_file_text(text_files)

                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
