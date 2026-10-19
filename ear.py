import os
import sys

# Configure Windows DLL directories for PyTorch CUDA, cuBLAS, and cuDNN
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["CT2_VERBOSE"] = "0"
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(torch_lib)
    except Exception:
        pass

os.environ["COQUI_TOS_AGREED"] = "1"
import queue
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import torch

# class Ear:
#     def __init__(self, model_size="base"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.compute_type = "float16" if self.device == "cuda" else "int8"
        
#         print(f"Loading Faster-Whisper ({model_size}) on: {self.device.upper()}...")
#         # 'base' model is tiny, ultra-fast (~150ms), and great for conversation
#         self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
#         print("Ear initialized successfully!")

#     def record_audio(self, filename="audio/input.wav", duration=5, sample_rate=16000):
#         """Records from the microphone for a set duration."""
#         os.makedirs("audio", exist_ok=True)
#         print("\n🎤 Listening... (Speak now)")
        
#         # Record audio from default microphone
#         audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
#         sd.wait() # Wait until recording is done
        
#         print("Recording finished. Transcribing...")
#         wav.write(filename, sample_rate, audio_data)
#         return filename

#     def listen_and_transcribe(self, duration=5) -> str:
#         """Records user speech and transcribes it into text with auto language detection."""
#         audio_path = self.record_audio(duration=duration)

#         # Guide Whisper to understand Indian slang & Tanglish in English letters
#         tanglish_hint = "Machi, bro, enna aachu, romba tired, sapdiya, pathukalam, tension aagadha"

#         segments, info = self.model.transcribe(
#             audio_path, 
#             beam_size=5,
#             initial_prompt=tanglish_hint
#         )
        
#         transcribed_text = " ".join([segment.text for segment in segments]).strip()
        
#         if transcribed_text:
#             print(f"You said: \"{transcribed_text}\"")
#         else:
#             print("No speech detected.")

#         return transcribed_text





# if __name__ == "__main__":
#     # Test script standalone
#     ear = Ear(model_size="base")
    
#     print("\n--- Testing Ear Module ---")
#     input("Press ENTER when you are ready to speak for 5 seconds...")
    
#     result = ear.listen_and_transcribe(duration=5)
#     print(f"\nFinal Result: {result}")

import speech_recognition as sr

class Ear:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust sensitivity for room noise
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        print("Ear initialized successfully (No cuDNN needed)!")

    def listen_and_transcribe(self, duration=5) -> str:
        """Listens from the laptop mic and converts speech to text."""
        with sr.Microphone() as source:
            print("\n🎤 Listening... Speak now (English/Tanglish)...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                # Listens until you stop speaking
                audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=8)
                print("Processing speech...")
                
                # Transcribes using Google's engine (free, no API key needed)
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
                print(f"Ear Error: {e}")
                return ""

if __name__ == "__main__":
    ear = Ear()
    print("Say something into your mic...")
    result = ear.listen_and_transcribe(duration=5)
    print(f"Result: {result}")