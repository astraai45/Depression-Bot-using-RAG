# Depression-Bot-using-RAG


Here’s a structured GitHub README for your project, including sections like Project Description, Installation, Usage, and Contributions.

````markdown
# Geriatric Depression Chatbot 🤖

This project provides a chatbot designed to assist elderly individuals who may be experiencing depression, loneliness, or emotional challenges. The chatbot leverages advanced Natural Language Processing (NLP) techniques, integrating multiple technologies such as Langchain, Twilio, and more, to offer compassionate and empathetic responses tailored to the user’s emotional state.

## Features
- **Voice Input:** The chatbot can accept and process voice-based queries.
- **Text Input:** Users can interact through text input for various queries.
- **Personalized Responses:** The bot responds with context-aware, personalized replies, especially for elderly users.
- **PDF Upload:** Users can upload PDF files (e.g., patient history) for more tailored responses.
- **Multi-language Support:** The bot supports multiple languages, including English, Telugu, and Tamil.
- **Negative Sentiment Detection:** The bot is capable of detecting negative sentiments and can send alerts to relatives using Twilio.

## Technologies Used
- **Streamlit** for the web interface.
- **Langchain** for building NLP models and handling various language tasks.
- **Ollama** for language models and embeddings.
- **Chroma** for vector storage and retrieval.
- **Twilio** for sending alerts to relatives.
- **PyPDF2 & PDFPlumber** for PDF parsing and document ingestion.
- **Deep Translator (Google Translate)** for multi-language support.
- **SpeechRecognition** for voice-to-text conversion.
- **Ragas** for emotional sentiment evaluation.

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/geriatric-depression-chatbot.git
cd geriatric-depression-chatbot
````

### Install Dependencies

Ensure that you have Python 3.8+ installed. Install the required Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

### Setup Twilio

1. Sign up at [Twilio](https://www.twilio.com/) and get your **Account SID**, **Auth Token**, and **Phone Number**.
2. Set up the Twilio credentials in the code by modifying the following variables:

   ```python
   TWILIO_ACCOUNT_SID = 'your_twilio_account_sid'
   TWILIO_AUTH_TOKEN = 'your_twilio_auth_token'
   TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'
   ```

### Setup the Language Model (Ollama)

Install the Ollama API by following the instructions on their official site or repository. You'll need to make sure you have access to Ollama’s API and models for embeddings.

## Usage

### Running the Chatbot

1. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```
2. Navigate to `http://localhost:8501` in your browser.
3. Interact with the chatbot by either typing or speaking your query.

### Uploading Patient History (PDF)

* You can upload a patient's PDF history via the file uploader in the sidebar, and the chatbot will tailor responses based on the context extracted from the PDF.

### Language Selection

* Choose from English, Telugu, or Tamil to receive responses in your preferred language.

### Voice Input

* Enable voice input from the sidebar and speak your query.

### Negative Sentiment Detection

* If negative words or phrases (e.g., "hopeless", "suicide", "alone") are detected in the user query, the system sends an alert to the specified relatives through Twilio.

## Contribution

Feel free to fork the repository and create a pull request if you would like to contribute to this project. Contributions are always welcome!

### Contributors

* [Pavan Balaji](https://github.com/pavanbalaji45)
* [Balaji Kartheek](https://github.com/Balaji-Kartheek)

