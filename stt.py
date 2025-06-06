
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cuda")

def transcribe(audio_path):
    segments, _ = model.transcribe(audio_path)
    full_text = " ".join([segment.text for segment in segments])
    return full_text
