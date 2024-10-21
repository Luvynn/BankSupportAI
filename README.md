# **Banking Customer Support System**

This repository implements a sophisticated chatbot system designed for banking customer support, featuring a user-friendly interface built with Streamlit, intent classification powered by sentence transformers, and advanced contextual response generation using the Gemini LLM. The system aims to enhance customer interaction by providing accurate and contextually relevant responses through both text and audio outputs.

## **Key Features:**

- **Speech Recognition:** Utilizes OpenAI Whisper for efficient transcription of user audio inputs.
- **Intent Classification:** Employs sentence embeddings and cosine similarity to classify user queries into predefined intents, ensuring accurate identification of user needs.
- **Contextual Response Generation:** Leverages Gemini LLM with integrated chat history memory to deliver informative and context-aware responses, enhancing customer experience.
- **Text-to-Speech Output:** Converts chatbot responses into audio using gTTS, allowing for seamless, natural interactions.

## **Installation:**

1. **Clone the Repository:**

```bash
git clone https://github.com/your-repo/banking-chatbot.git
```

2. **Create a Virtual Environment (Recommended):**

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set Up API Key:**

Add your OpenAI API key to a `.env` file:

```bash
OPENAI_API_KEY="your_openai_api_key"
```

## **Usage:**

1. **Run the Application:**

```bash
streamlit run app.py
```

2. **Interact with the Chatbot:**
   - Type your question or message in the provided text field.
   - Optionally, click the microphone icon to record your audio query.

## **Code Structure:**

- **`app.py`:** Manages the main application logic, handles user input through the Streamlit interface, and coordinates chatbot interactions.
- **`chatbot.py`:** Defines the `ChatBot` class responsible for generating responses via LLMChain and classifying intents using sentence transformers.
- **`audio_utils.py`:** Provides functionality for transcribing audio (using OpenAI Whisper), converting text to speech (using gTTS), and playing audio responses in the Streamlit app.

## **Technical Details:**

### **Intent Classification:**
- Sentence embeddings are generated using a pre-trained sentence transformer model for both user inputs and predefined intents.
- Cosine similarity is calculated between the embeddings of the user query and intent dataset to determine the most relevant intent.

### **Chatbot Response Generation:**
- A `ChatBot` instance integrates an `LLMChain` object with a custom prompt template.
- The system uses the chat history and user query to generate a context-aware response, aligned with the classified intent.

### **Audio Processing:**
- **Speech Transcription:** OpenAI Whisper is used to convert speech input into text.
- **Text-to-Speech Conversion:** The `gTTS` library generates audio responses, saving them as WAV files, which are then played back within the Streamlit app for seamless interaction.

