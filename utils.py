import sounddevice as sd
import soundfile as sf


def record_audio(filename, duration=5, samplerate=16000):
    print("Recording...")
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
    )
    sd.wait()
    sf.write(filename, audio, samplerate)
    print("Recording finished.")


def play_audio(filename):
    data, samplerate = sf.read(filename)
    sd.play(data, samplerate)
    sd.wait()
