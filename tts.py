import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig

# Allowlist XttsConfig to avoid UnpicklingError
add_safe_globals([XttsConfig])

from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

def text_to_speech(text, output_path="output.wav"):
    tts.tts_to_file(text=text, speaker_wav=None, language="en", file_path=output_path)
