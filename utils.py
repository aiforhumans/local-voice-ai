"""Audio utility functions for recording and playback."""

from __future__ import annotations

import sounddevice as sd
import soundfile as sf


def record_audio(filename: str, duration: int = 5, samplerate: int = 16000) -> None:
    """Record audio from the microphone and save it as ``filename``."""

    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, audio, samplerate)
    print("Recording finished.")


def play_audio(filename: str) -> None:
    """Play an audio file through the speakers."""

    data, samplerate = sf.read(filename)
    sd.play(data, samplerate)
    sd.wait()
