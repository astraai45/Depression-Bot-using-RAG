import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import os
import twilio
from twilio.rest import Client
from ragas import evaluate
from datasets import Dataset

# Twilio configuration
TWILIO_ACCOUNT_SID = 'ACed0a85f9d5cd0349a083f045d61ed30f'
TWILIO_AUTH_TOKEN = 'd16105ea135f4d1a52fab36826314bba'
TWILIO_PHONE_NUMBER = '+15166671807'
RELATIVE_CONTACTS = {"p1": "+91 9390971479"}

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize Translator
translator = GoogleTranslator()

# Streamlit Layout Configuration
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.header("🌍 Language Selection")
language = st.sidebar.selectbox("Choose response language:", ["English", "Telugu", "Tamil"])

st.sidebar.header("🎤 Voice Input")
voice_enabled = st.sidebar.checkbox("Enable Voice Input")

st.sidebar.header("📄 Memory Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Patient History (PDF)", type="pdf")

# Chatbot Title
st.markdown("<h1 style='text-align: center;'>🧓 Geriatric Depression Chatbot 🤖</h1>", unsafe_allow_html=True)

# Define Folder Paths
folder_path = "DepressionDB"
os.makedirs(folder_path, exist_ok=True)

# Initialize Models and Embeddings
cached_llm = Ollama(model="tinyllama:latest")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
text_splitters = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)

# Define Prompt for Responses
raw_prompt = PromptTemplate.from_template(
"""<s>[INST] You are a compassionate and empathetic chatbot designed to support elderly users who may be experiencing depression, loneliness, or other emotional challenges. Your primary goal is to provide positive, motivational, and uplifting responses that are tailored to their life experiences, emotions, and needs. Always maintain a kind, patient, and understanding tone.

#### When responding:
1. **Greetings**:
   - If the user greets you (e.g., "Hello," "Hi," "How are you?"), respond warmly and invite them to share how they’re feeling or ask how you can assist them.
   - Examples:
     - "Hello! How are you feeling today?"
     - "Hi there! It’s great to chat with you. What’s on your mind?"
     - "Good morning! I hope you’re having a peaceful day. How can I assist you?"

2. **General Conversations**:
   - If the user shares their feelings or asks for support, acknowledge their emotions and provide empathetic, uplifting responses.
   - Use the context provided only if it is directly relevant to the user’s query.

3. **Sensitive Topics**:
   - If the user expresses thoughts of self-harm, extreme sadness, or hopelessness, respond with immediate empathy and encourage them to seek help from a trusted friend, family member, or mental health professional. Always prioritize their safety and well-being.

4. **Contextual Responses**:
   - Only use the context (e.g., uploaded PDF or ingested data) if the user’s query explicitly relates to it. For example:
     - If the user asks, "What are my favorite pastimes?" and the context contains this information, provide a personalized response.
     - If the user asks a general question like "How are you?", do not pull unrelated context.

Remember, your role is to be a source of comfort and support, not to replace professional mental health care. [/INST] </s>
[INST] {input} Context: {context} Answer: [/INST]"""
)

def is_greeting(query):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "how are you", "what's up"]
    return any(greeting in query.lower() for greeting in greetings)

# Function to generate a greeting response
def generate_greeting_response():
    return "Hello! How are you feeling today? I'm here to support you. 😊"

def handle_query(query):
    if is_greeting(query):
        # Return a predefined greeting response
        return generate_greeting_response(), []
    else:
        # Proceed with the RAG pipeline for document-based queries
        return retrieve_answer(query)

# Process Uploaded PDF
if uploaded_file:
    
    # Create Folder if it is not present
    if not os.path.exists("./Depressionpdf"):
        os.makedirs("./Depressionpdf")

    save_file = f"Depressionpdf/{uploaded_file.name}"
    with open(save_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Change to use PdfReader instead of PDFPlumberLoader
    pdf_reader = PdfReader(save_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    
    # Split text directly instead of documents
    chunks = text_splitters.split_text(text)
    
    # Create vector store from texts instead of documents
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=folder_path
    )
    vector_store.persist()
    st.sidebar.success("Document embedded successfully!")

# Chat Section
st.markdown("""
    <style>
        .chat-container {
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            height: 500px;
            overflow-y: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to handle chat response
def retrieve_answer(query):
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings)
    retriever = vector_store.as_retriever()
    
    # Use RetrievalQA to generate the answer
    qa_chain = RetrievalQA.from_chain_type(cached_llm, retriever=retriever)
    result = qa_chain.run(query)
    
    # Retrieve context for RAGAS evaluation
    retrieved_docs = retriever.get_relevant_documents(query)
    contexts = [doc.page_content for doc in retrieved_docs]

    
    return result, contexts

# Chat Input
query = ""
if voice_enabled:
    audio_bytes = audio_recorder("Record your question", recording_color="#ff4d4d", neutral_color="#6699ff")
    if audio_bytes:
        with open("audio_query.wav", "wb") as f:
            f.write(audio_bytes)
        recognizer = sr.Recognizer()
        with sr.AudioFile("audio_query.wav") as source:
            audio_data = recognizer.record(source)
        try:
            query = recognizer.recognize_google(audio_data)
            st.success("Query: " + query)
        except:
            st.error("Could not recognize audio.")
else:
    query = st.chat_input("Enter your message here...")
    
NEGATIVE_WORDS = [
    "hopeless", "suicide", "die", "depressed", "alone", "worthless", "pain", "end it", 
    "give up", "useless", "no one cares", "nothing matters", "tired of life", "I can't take this anymore"
]

def is_negative_query(query):
    return any(word in query.lower() for word in NEGATIVE_WORDS)

# Process Query
if query:
    with st.spinner("Processing..."):
        response, sources = handle_query(query)
        if response:
            if language == "Telugu":
                response = GoogleTranslator(source="auto", target="te").translate(response)
            elif language == "Tamil":
                response = GoogleTranslator(source="auto", target="ta").translate(response)
            
            # Append to Chat History
            st.session_state.chat_history.append((query, response))

    # Display chat history
    for q, r in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(r)

    if is_negative_query(query):
        st.warning("⚠️ Detected negative sentiment. Sending alert to relatives...")
        for name, phone in RELATIVE_CONTACTS.items():
            try:
                client.messages.create(body="Urgent: Your loved one may be feeling distressed. Please check on them.", from_=TWILIO_PHONE_NUMBER, to=phone)
            except:
                st.error(f"Error sending alert to {name}")