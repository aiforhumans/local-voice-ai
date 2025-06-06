
# Local Voice AI Companion

🎤 Talk with a Local AI model (LM Studio) with voice.

## Features
- 🎙️ Speech to Text (Whisper)
- 🤖 LLM chat (LM Studio)
- 🔊 Text to Speech (XTTS)
- 🖥️ Gradio UI

## Setup

### 1️⃣ Setup Python env

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Run LM Studio
- Start LM Studio
- Load a model (chat model, e.g. LLaMA3)

### 3️⃣ Run the App

```bash
python app.py
```

## Notes
- Works great with RTX GPU
- You can tune Whisper + XTTS language
- Easily extend with memory

Enjoy 🚀
