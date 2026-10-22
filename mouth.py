import os
import sys
import re
import queue
import threading
import sounddevice as sd
import soundfile as sf
import torch
import config

# Windows console encoding helper
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["COQUI_TOS_AGREED"] = "1"
# Disable cuDNN to prevent Windows DLL / status mismatch errors on RTX 3050 Laptop GPU
torch.backends.cudnn.enabled = False

class Mouth:
    def __init__(self, voice_sample_path: str = None):
        self.voice_sample = voice_sample_path or config.DEFAULT_VOICE_SAMPLE
        self.output_dir = config.AUDIO_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        self._validate_voice_sample()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Mouth] Loading XTTS-v2 on: {self.device.upper()}...")
        from TTS.api import TTS
        self.tts = TTS(model_name=config.TTS_MODEL_NAME).to(self.device)
        print("[Mouth] Initialized with pipelined audio playback!")

    def _validate_voice_sample(self):
        """Ensures a valid voice sample exists, checking common filename variations."""
        if os.path.exists(self.voice_sample):
            return

        # Check for case variations
        directory = os.path.dirname(self.voice_sample) or config.VOICE_SAMPLES_DIR
        basename = os.path.basename(self.voice_sample).lower()
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.lower() == basename:
                    self.voice_sample = os.path.join(directory, file)
                    return

        # Fallback to any .wav in voice_samples
        if os.path.exists(config.VOICE_SAMPLES_DIR):
            samples = [os.path.join(config.VOICE_SAMPLES_DIR, f) for f in os.listdir(config.VOICE_SAMPLES_DIR) if f.endswith(".wav")]
            if samples:
                self.voice_sample = samples[0]
                print(f"[Warning] Voice sample fallback used: {self.voice_sample}")
                return

        print(f"[Warning] Voice sample not found at {self.voice_sample}")

    def _normalize_for_speech(self, text: str) -> str:
        """Removes symbols, emojis, and normalizes spoken text."""
        # Strip emojis and markdown
        text = re.sub(r'[*#_~`>]', '', text)
        text = re.sub(r'[\(\[\{].*?[\)\]\}]', '', text)
        # Remove stray punctuation that causes XTTS to stutter
        text = re.sub(r'--+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _split_into_sentences(self, text: str):
        """Splits full response into natural spoken sentences."""
        clean = self._normalize_for_speech(text)
        sentences = re.split(r'(?<=[.!?])\s+', clean)
        return [s.strip() for s in sentences if len(s.strip()) > 1]

    def _play_worker(self, play_queue: queue.Queue):
        """Background thread worker to play audio files as they are ready."""
        while True:
            audio_path = play_queue.get()
            if audio_path is None:  # Sentinel to stop
                play_queue.task_done()
                break
            try:
                data, fs = sf.read(audio_path)
                sd.play(data, fs)
                sd.wait()
            except Exception as e:
                print(f"[Playback Error]: {e}")
            finally:
                play_queue.task_done()

    def speak_stream(self, sentence_generator, language: str = "en"):
        """
        Pipelined generation & playback:
        Plays sentence N while generating sentence N+1 in parallel!
        """
        play_queue = queue.Queue()
        worker = threading.Thread(target=self._play_worker, args=(play_queue,), daemon=True)
        worker.start()

        chunk_idx = 0
        try:
            for sentence in sentence_generator:
                clean_sentence = self._normalize_for_speech(sentence)
                if not clean_sentence:
                    continue

                print(f"\n[Jarvis Speaking]: \"{clean_sentence}\"")
                chunk_file = os.path.join(self.output_dir, f"chunk_{chunk_idx % 10}.wav")
                chunk_idx += 1

                # Generate audio
                self.tts.tts_to_file(
                    text=clean_sentence,
                    speaker_wav=self.voice_sample,
                    language=language,
                    file_path=chunk_file,
                    speed=config.TTS_SPEED,
                    temperature=config.TTS_TEMPERATURE,
                    repetition_penalty=config.TTS_REPETITION_PENALTY
                )

                # Queue audio for immediate playback in background thread
                play_queue.put(chunk_file)

        finally:
            # Send sentinel and wait for playback to finish
            play_queue.put(None)
            play_queue.join()
            worker.join()

    def speak(self, text: str, language: str = "en"):
        """Speaks text using sentence splitting with pipelined playback."""
        if not text.strip():
            return
        sentences = self._split_into_sentences(text)
        self.speak_stream(sentences, language=language)

if __name__ == "__main__":
    speaker = Mouth()
    test_message = "Hey machi, I am back. How are you doing today? Let us build something cool together."
    speaker.speak(test_message, language="en")