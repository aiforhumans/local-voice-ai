import os
import sys
from unittest.mock import MagicMock, patch
import types

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Create dummy modules for missing dependencies
faster_whisper_mock = types.ModuleType("faster_whisper")
faster_whisper_mock.WhisperModel = MagicMock()
sys.modules["faster_whisper"] = faster_whisper_mock

requests_mock = types.ModuleType("requests")
requests_mock.post = MagicMock()
sys.modules["requests"] = requests_mock

sounddevice_mock = types.ModuleType("sounddevice")
sounddevice_mock.rec = MagicMock()
sounddevice_mock.wait = MagicMock()
sounddevice_mock.play = MagicMock()
sys.modules["sounddevice"] = sounddevice_mock

soundfile_mock = types.ModuleType("soundfile")
soundfile_mock.write = MagicMock()
soundfile_mock.read = MagicMock(return_value=("data", 123))
sys.modules["soundfile"] = soundfile_mock

torch_mock = types.ModuleType("torch")
serialization_module = types.ModuleType("torch.serialization")
serialization_module.add_safe_globals = MagicMock()
torch_mock.serialization = serialization_module
torch_mock.cuda = types.SimpleNamespace(is_available=MagicMock(return_value=False))
sys.modules["torch"] = torch_mock
sys.modules["torch.serialization"] = serialization_module

shared_configs_module = types.ModuleType("TTS.config.shared_configs")
shared_configs_module.BaseDatasetConfig = MagicMock()
sys.modules["TTS.config.shared_configs"] = shared_configs_module

TTS_module = types.ModuleType("TTS")
sys.modules["TTS"] = TTS_module
TTS_tts = types.ModuleType("TTS.tts")
sys.modules["TTS.tts"] = TTS_tts
configs_module = types.ModuleType("TTS.tts.configs.xtts_config")
configs_module.XttsConfig = MagicMock()
sys.modules["TTS.tts.configs.xtts_config"] = configs_module
models_module = types.ModuleType("TTS.tts.models.xtts")
models_module.XttsAudioConfig = MagicMock()
models_module.XttsArgs = MagicMock()
sys.modules["TTS.tts.models.xtts"] = models_module
api_module = types.ModuleType("TTS.api")
api_module.TTS = MagicMock()
sys.modules["TTS.api"] = api_module

import stt  # noqa: E402
import chat  # noqa: E402
import tts  # noqa: E402
import utils  # noqa: E402


class DummyResponse:
    def __init__(self, json_data):
        self._json = json_data

    def json(self):
        return self._json


@patch("stt.model")
def test_transcribe(mock_model):
    mock_segment = MagicMock()
    mock_segment.text = "hello"
    mock_model.transcribe.return_value = ([mock_segment], None)
    assert stt.transcribe("dummy.wav") == "hello"


@patch("requests.post")
def test_chat_with_lmstudio(mock_post):
    mock_post.return_value = DummyResponse(
        {"choices": [{"message": {"content": "hi"}}]}
    )
    assert chat.chat_with_lmstudio("hello") == "hi"


@patch("tts.tts")
def test_text_to_speech(mock_tts):
    tts.text_to_speech("hello", "out.wav")
    mock_tts.tts_to_file.assert_called_with(
        text="hello", speaker_wav=None, language="en", file_path="out.wav"
    )


@patch("utils.sd")
@patch("utils.sf.write")
def test_record_audio(mock_write, mock_sd):
    utils.record_audio("in.wav", duration=1)
    mock_sd.rec.assert_called()
    mock_write.assert_called_with("in.wav", mock_sd.rec.return_value, 16000)


@patch("utils.sd")
@patch("utils.sf.read")
def test_play_audio(mock_read, mock_sd):
    mock_read.return_value = ("data", 123)
    utils.play_audio("in.wav")
    mock_read.assert_called_with("in.wav")
    mock_sd.play.assert_called_with("data", 123)
