import gradio as gr
from stt import transcribe
from chat import chat_with_lmstudio
from tts import text_to_speech
from utils import record_audio, play_audio


def voice_chat():
    record_audio("input.wav", duration=5)
    user_text = transcribe("input.wav")
    print(f"User said: {user_text}")

    reply_text = chat_with_lmstudio(user_text)
    print(f"AI reply: {reply_text}")

    text_to_speech(reply_text, "output.wav")
    play_audio("output.wav")

    return user_text, reply_text


with gr.Blocks() as demo:
    gr.Markdown("## 🎤 Local Voice AI Companion")
    btn = gr.Button("🎙️ Talk to AI")
    user_textbox = gr.Textbox(label="You said")
    ai_textbox = gr.Textbox(label="AI replied")

    btn.click(voice_chat, outputs=[user_textbox, ai_textbox])

demo.launch()
