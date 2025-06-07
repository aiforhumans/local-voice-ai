from torch.serialization import add_safe_globals
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

# Allowlist required config classes to avoid UnpicklingError
add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
).to(device)


def text_to_speech(text, output_path="output.wav"):
    tts.tts_to_file(
        text=text,
        speaker_wav=None,
        language="en",
        file_path=output_path,
    )
