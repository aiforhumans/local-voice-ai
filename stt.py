"""Speech-to-text utilities using Faster Whisper."""

from __future__ import annotations

import torch
from faster_whisper import WhisperModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model() -> WhisperModel:
    """Load and return the Whisper model."""

    return WhisperModel("medium", device=DEVICE)


# Keep a module-level reference so tests can patch it easily
model = _load_model()


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file into text."""

    segments, _ = model.transcribe(audio_path)
    texts = [seg.text.strip() for seg in segments if seg.text]
    return " ".join(texts)
