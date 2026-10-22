import os
import sys
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import config

# Windows console encoding and DLL helpers
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["CT2_VERBOSE"] = "0"

class Ear:
    def __init__(self, engine: str = None, model_size: str = None):
        self.engine = engine or config.STT_ENGINE
        self.model_size = model_size or config.WHISPER_MODEL_SIZE
        self.tanglish_hint = config.TANGLISH_HINT
        self.audio_dir = config.AUDIO_DIR
        os.makedirs(self.audio_dir, exist_ok=True)
        self.input_wav = os.path.join(self.audio_dir, "input.wav")

        self.whisper_model = None
        self.recognizer = None

        if self.engine == "whisper":
            self._init_whisper()

        # If whisper fails or google was selected, init speech_recognition
        if self.whisper_model is None:
            self._init_google()

    def _init_whisper(self):
        """Initializes Faster-Whisper on CPU int8 to save RTX 3050 VRAM and avoid cuDNN conflicts."""
        try:
            from faster_whisper import WhisperModel
            print(f"[Ear] Loading Faster-Whisper ({self.model_size}) on CPU (int8)...")
            self.whisper_model = WhisperModel(
                self.model_size,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE
            )
            print("[Ear] Initialized with Faster-Whisper (Tanglish optimized)!")
        except Exception as e:
            print(f"[Warning] Faster-Whisper initialization failed ({e}). Falling back to Google STT...")
            self.whisper_model = None

    def _init_google(self):
        """Initializes speech_recognition as a resilient fallback."""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            print("[Ear] Initialized with Google STT engine.")
        except Exception as e:
            print(f"[Warning] Failed to initialize SpeechRecognition: {e}")

    def record_audio(self, duration: int = None, sample_rate: int = 16000) -> str:
        """Records from the default microphone for a set duration."""
        duration = duration or config.LISTEN_DURATION
        print(f"\n[Ear] Listening... (Speak now for up to {duration}s)")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        wav.write(self.input_wav, sample_rate, audio_data)
        return self.input_wav

    def listen_and_transcribe(self, duration: int = None) -> str:
        """Records and transcribes user speech using Whisper or Google STT."""
        duration = duration or config.LISTEN_DURATION

        # 1. Primary: Faster-Whisper with Tanglish initial prompt
        if self.whisper_model is not None:
            try:
                audio_path = self.record_audio(duration=duration)
                print("[Ear] Processing speech...")
                segments, info = self.whisper_model.transcribe(
                    audio_path,
                    beam_size=5,
                    initial_prompt=self.tanglish_hint,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                transcribed = " ".join([segment.text for segment in segments]).strip()
                if transcribed:
                    print(f"You said: \"{transcribed}\"")
                else:
                    print("No speech detected.")
                return transcribed
            except Exception as e:
                print(f"[Warning] Whisper error ({e}), trying fallback...")

        # 2. Fallback: Google Speech Recognition
        if self.recognizer is not None:
            import speech_recognition as sr
            with sr.Microphone() as source:
                print("\n[Ear] Listening... Speak now...")
                try:
                    audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=config.PHRASE_TIME_LIMIT)
                    print("[Ear] Processing speech...")
                    text = self.recognizer.recognize_google(audio, language="en-IN")
                    print(f"You said: \"{text}\"")
                    return text
                except sr.WaitTimeoutError:
                    print("No speech detected within time limit.")
                    return ""
                except sr.UnknownValueError:
                    print("Couldn't understand the audio clearly.")
                    return ""
                except Exception as e:
                    print(f"[Ear Error]: {e}")
                    return ""

        return ""

if __name__ == "__main__":
    ear = Ear()
    print("Say something into your mic...")
    result = ear.listen_and_transcribe(duration=5)
    print(f"Result: {result}")