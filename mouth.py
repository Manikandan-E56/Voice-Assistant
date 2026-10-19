import os
os.environ["COQUI_TOS_AGREED"] = "1"
import torch
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS

class Mouth:
    def __init__(self, voice_sample_path="voice_samples/induja_Voice.wav"):
        self.voice_sample = voice_sample_path
        self.output_path = "audio/output.wav"
        
        # Ensure audio output folder exists
        os.makedirs("audio", exist_ok=True)

        if not os.path.exists(self.voice_sample):
            print(f"⚠️ Warning: Voice sample not found at {self.voice_sample}")

        # Check if CUDA (NVIDIA GPU) is available on your ROG G15
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading XTTS-v2 on: {self.device.upper()}...")

        # Initialize the zero-shot voice cloning model
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        print("Mouth initialized successfully!")

    def speak(self, text: str, language: str = "en"):
        """Clones your voice and plays the audio directly through speakers."""
        if not text.strip():
            return

        print(f"\n[Jarvis Speaking]: \"{text}\"")
        
        # Generate cloned audio
        # Optimized generation settings
        self.tts.tts_to_file(
            text=text,
            speaker_wav=self.voice_sample,
            language=language,
            file_path=self.output_path,
            speed=1.1,                 # ⚡ Makes speech snappy & conversational
            temperature=0.65,          # 🎯 Balances emotion & clarity
            repetition_penalty=2.0     # 🚫 Prevents voice glitching
        )

        # Play audio through speakers
        data, fs = sf.read(self.output_path)
        sd.play(data, fs)
        sd.wait() # Wait until finished speaking

if __name__ == "__main__":
    # Test script standalone
    speaker = Mouth(voice_sample_path="voice_samples/induja_Voice.wav")
    
    test_message = "Hey bro! I'm speaking with your own voice now. How do I sound?"
    speaker.speak(test_message, language="en")