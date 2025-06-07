from faster_whisper import WhisperModel
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperModel("medium", device=device)


def transcribe(audio_path):
    segments, _ = model.transcribe(audio_path)
    full_text = " ".join([segment.text for segment in segments])
    return full_text
