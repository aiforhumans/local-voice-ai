"""Text-to-speech helper using XTTS."""

from __future__ import annotations

import torch
from torch.serialization import add_safe_globals
from TTS.api import TTS
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs

# Allowlist necessary configs for safe torch unpickling
add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Keep a module-level object to allow easy mocking in tests
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)


def text_to_speech(text: str, output_path: str = "output.wav") -> None:
    """Convert ``text`` to spoken audio saved at ``output_path``."""

    tts.tts_to_file(text=text, speaker_wav=None, language="en", file_path=output_path)
