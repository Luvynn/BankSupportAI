Perfect 👍 — that’s already a solid README. Let’s make it **more polished and recruiter-friendly** under a new repo name (e.g., **`BankSupportAI`**).
Here’s the cleaned-up version you can drop directly into `README.md`:

---

# BankSupportAI

An intelligent chatbot system for **banking customer support**, built with **Streamlit**, **OpenAI Whisper**, **Sentence Transformers**, and the **Gemini LLM**.  
The system enhances customer interactions by providing accurate, contextually relevant responses through both **text** and **audio** outputs.

---

## 🚀 Key Features
- **Speech Recognition** – transcribe user audio inputs with OpenAI Whisper  
- **Intent Classification** – classify queries using sentence embeddings + cosine similarity  
- **Contextual Response Generation** – generate informed replies with Gemini LLM and chat history memory  
- **Text-to-Speech Output** – convert chatbot responses into natural audio via gTTS  

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/BankSupportAI.git
cd BankSupportAI
````

2. **Create a virtual environment (recommended)**

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up API key**
   Add your OpenAI API key to a `.env` file:

```bash
OPENAI_API_KEY="your_openai_api_key"
```

---

## ▶️ Usage

1. **Run the application**

```bash
streamlit run app.py
```

2. **Interact with the chatbot**

* Type your question or message in the input field
* Optionally, click the microphone icon to record an audio query

---

## 📂 Code Structure

* **`app.py`** – main app logic, Streamlit UI, chatbot coordination
* **`chatbot.py`** – defines the `ChatBot` class with LLMChain and intent classification
* **`audio_utils.py`** – handles transcription (Whisper), text-to-speech (gTTS), and audio playback
* **`intent.csv` / `intent_embeddings.csv`** – predefined intents and embeddings dataset

---

## 🔍 Technical Details

### Intent Classification

* Uses sentence transformer embeddings for both queries and intents
* Cosine similarity selects the most relevant intent

### Chatbot Response Generation

* `ChatBot` integrates with `LLMChain` and a custom prompt template
* Responses are context-aware, leveraging chat history + intent classification

### Audio Processing

* **Transcription:** OpenAI Whisper converts audio → text
* **Text-to-Speech:** gTTS generates WAV responses, played in Streamlit

---
