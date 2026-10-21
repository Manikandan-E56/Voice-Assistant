import os
os.environ["COQUI_TOS_AGREED"] = "1"
import re
import torch
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS

# class Mouth:
#     def __init__(self, voice_sample_path="voice_samples/induja_Voice.wav"):
#         self.voice_sample = voice_sample_path
#         self.output_path = "audio/output.wav"
        
#         # Ensure audio output folder exists
#         os.makedirs("audio", exist_ok=True)

#         if not os.path.exists(self.voice_sample):
#             print(f"⚠️ Warning: Voice sample not found at {self.voice_sample}")

#         # Check if CUDA (NVIDIA GPU) is available on your ROG G15
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"Loading XTTS-v2 on: {self.device.upper()}...")

#         # Initialize the zero-shot voice cloning model
#         self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
#         print("Mouth initialized successfully!")

#     def speak(self, text: str, language: str = "en"):
#         """Clones your voice and plays the audio directly through speakers."""
#         if not text.strip():
#             return

#         print(f"\n[Jarvis Speaking]: \"{text}\"")
        
#         # Generate cloned audio
#         # Optimized generation settings
#         self.tts.tts_to_file(
#             text=text,
#             speaker_wav=self.voice_sample,
#             language=language,
#             file_path=self.output_path,
#             speed=1.1,                 # ⚡ Makes speech snappy & conversational
#             temperature=0.65,          # 🎯 Balances emotion & clarity
#             repetition_penalty=2.0     # 🚫 Prevents voice glitching
#         )

#         # Play audio through speakers
#         data, fs = sf.read(self.output_path)
#         sd.play(data, fs)
#         sd.wait() # Wait until finished speaking



torch.backends.cudnn.enabled = False

class Mouth:
    def __init__(self, voice_sample_path="voice_samples/induja_Voice.wav"):
        self.voice_sample = voice_sample_path
        self.output_path = "audio/chunk.wav"
        os.makedirs("audio", exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading XTTS-v2 on: {self.device.upper()}...")
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        print("Mouth initialized with speed optimizations!")

    def _split_into_sentences(self, text: str):
        """Splits full response into natural spoken sentences."""
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def speak(self, text: str, language: str = "en"):
        """Speaks sentence-by-sentence for fast response time."""
        if not text.strip():
            return

        sentences = self._split_into_sentences(text)
        
        for idx, sentence in enumerate(sentences):
            print(f"[Speaking]: \"{sentence}\"")
            chunk_file = f"audio/chunk_{idx}.wav"
            
            # Generate speech for this sentence
            self.tts.tts_to_file(
                text=sentence,
                speaker_wav=self.voice_sample,
                language=language,
                file_path=chunk_file,
                speed=1.1,
                temperature=0.65,
                repetition_penalty=2.0
            )

            # Play audio immediately
            data, fs = sf.read(chunk_file)
            sd.play(data, fs)
            sd.wait()


if __name__ == "__main__":
    # Test script standalone
    speaker = Mouth(voice_sample_path="voice_samples/induja_Voice.wav")
    
    test_message = """Hey, I'm back. How are you? I hope you are doing well. I was just relaxing and chilling. Now I'm feeling a bit tired and sleepy. I need to take some rest and recharge myself. I will be back soon with more energy and enthusiasm. Take care of yourself and have a great day ahead. See you later, macha."""
    speaker.speak(test_message, language="en")