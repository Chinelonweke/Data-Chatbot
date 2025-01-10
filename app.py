import streamlit as st
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
import hashlib  


# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Function to authenticate user
def authenticate_user(username, password, user_db):
    hashed_password = hash_password(password)
    return username in user_db and user_db[username]["password"] == hashed_password

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


#process the conversation, and display chat history
def handle_userinput(user_question):
    # Process the user's input and fetch the conversation response
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the conversation with clear structure
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User's message
            # Display the user's message with a question icon
            st.write(f"**🤖❓ User:** {message.content}")
        else:  # Bot's response
            # Display the bot's message with a bot icon
            st.write(f"**🤖 Bot:** {message.content}")








# Main function
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs, Docs, and Texts",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = {}  # To store user-specific data

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if not st.session_state.conversation:
        with st.sidebar:
            st.header("Login / Register")
            user_action = st.radio("Choose Action", ["Login", "Register"])

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if user_action == "Register":
                if st.button("Register"):
                    if username in st.session_state.user_data:
                        st.warning("Username already exists. Please choose another.")
                    else:
                        st.session_state.user_data[username] = {"password": hash_password(password), "data": None}
                        st.success("Registration successful! Please log in.")

            elif user_action == "Login":
                if st.button("Login"):
                    if authenticate_user(username, password):
                        st.session_state.conversation = True
                        st.session_state.current_user = username
                        st.success(f"Welcome, {username}!")
                    else:
                        st.error("Invalid username or password.")
    else:
        st.header("Chat with PDFs, Word Docs, and Text Files :books:")
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Your documents")
            uploaded_files = st.file_uploader(
                "Upload your PDFs, Word docs, and text files here", accept_multiple_files=True)

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


if __name__ == '__main__':
    main()
