"""Gradio application wiring together STT, LLM and TTS."""

from __future__ import annotations

import gradio as gr

from stt import transcribe
from chat import chat_with_lmstudio
from tts import text_to_speech
from utils import record_audio, play_audio


INPUT_FILE = "input.wav"
OUTPUT_FILE = "output.wav"


def voice_chat() -> tuple[str, str]:
    """Perform one voice interaction cycle."""

    record_audio(INPUT_FILE, duration=5)
    user_text = transcribe(INPUT_FILE)
    print(f"User said: {user_text}")

    reply_text = chat_with_lmstudio(user_text)
    print(f"AI reply: {reply_text}")

    text_to_speech(reply_text, OUTPUT_FILE)
    play_audio(OUTPUT_FILE)

    return user_text, reply_text


def main() -> None:
    with gr.Blocks() as demo:
        gr.Markdown("## 🎤 Local Voice AI Companion")
        btn = gr.Button("🎙️ Talk to AI")
        user_box = gr.Textbox(label="You said")
        ai_box = gr.Textbox(label="AI replied")
        btn.click(voice_chat, outputs=[user_box, ai_box])
    demo.launch()


if __name__ == "__main__":
    main()
