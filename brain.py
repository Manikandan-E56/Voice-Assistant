import os
import sys
import re
import ollama
from typing import Generator, List, Dict
import config

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

class Brain:
    def __init__(self, model_name: str = None):
        self.primary_model = model_name or config.PRIMARY_LLM_MODEL
        self.fallback_model = config.FALLBACK_LLM_MODEL
        self.model_name = self._resolve_model()
        self.options = config.LLM_OPTIONS
        self.max_history = config.MAX_HISTORY_TURNS

        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": config.SYSTEM_PROMPT}
        ]
        print(f"[Brain] Loaded with model: {self.model_name}")

    def _get_available_models(self) -> List[str]:
        models_response = ollama.list()
        available = []
        if isinstance(models_response, dict) and "models" in models_response:
            available = [m.get("model") or m.get("name") for m in models_response["models"]]
        elif hasattr(models_response, "models"):
            available = [getattr(m, "model", getattr(m, "name", "")) for m in models_response.models]
        return available

    def _resolve_model(self) -> str:
        """Checks if primary model exists in local Ollama; starts Ollama if offline and falls back gracefully."""
        try:
            available = self._get_available_models()
        except Exception:
            print("[Brain] Ollama is not running. Attempting to start 'ollama serve' in background...")
            try:
                import subprocess
                import time
                creationflags = 0x00000008 if sys.platform == "win32" else 0  # DETACHED_PROCESS
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags
                )
                time.sleep(3)
                available = self._get_available_models()
            except Exception as e:
                print(f"[Brain Warning] Could not connect to Ollama server ({e}).")
                print("[Brain Tip] Please make sure Ollama is installed and running ('ollama serve' or open Ollama app).")
                return self.primary_model

        for m in available:
            if self.primary_model in m:
                return self.primary_model

        for m in available:
            if self.fallback_model in m:
                print(f"[Warning] Primary model '{self.primary_model}' not found in Ollama. Using fallback '{self.fallback_model}'.")
                return self.fallback_model

        return self.primary_model

    def _clean_text_for_speech(self, text: str) -> str:
        """Strips asterisks, markdown syntax, action tags, and excess whitespace."""
        cleaned = re.sub(r'[*_#`]', '', text)
        cleaned = re.sub(r'[\(\[\{].*?[\)\]\}]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _prune_history(self):
        """Maintains a sliding window so context memory doesn't grow indefinitely."""
        if len(self.conversation_history) > (self.max_history * 2 + 1):
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-(self.max_history * 2):]

    def _build_payload(self, user_input: str, memory_context: str = "") -> List[Dict[str, str]]:
        """
        Builds the messages array for Ollama.
        Injects memory_context as an ephemeral system note so the user's role is not polluted.
        """
        payload = list(self.conversation_history)
        if memory_context:
            ephemeral_note = {
                "role": "system",
                "content": f"[RECALLED USER CONTEXT (use subtly if relevant, do not recite verbatim)]: {memory_context}"
            }
            payload.append(ephemeral_note)

        payload.append({"role": "user", "content": user_input})
        return payload

    def stream_sentences(self, user_input: str, memory_context: str = "") -> Generator[str, None, str]:
        """
        Streams Ollama tokens and yields complete sentences as they are formed.
        Allows TTS to begin speaking sentence 1 while sentence 2 is being generated!
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        self._prune_history()

        payload = self._build_payload(user_input, memory_context)
        sentence_buffer = ""
        full_reply = ""

        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=payload,
                options=self.options,
                stream=True
            )

            sentence_end_regex = re.compile(r'([.!?\n]+)')

            for chunk in stream:
                token = chunk['message']['content']
                print(token, end="", flush=True)
                full_reply += token
                sentence_buffer += token

                parts = sentence_end_regex.split(sentence_buffer)
                if len(parts) > 2:
                    for i in range(0, len(parts) - 1, 2):
                        sentence_text = parts[i] + parts[i + 1]
                        clean_sentence = self._clean_text_for_speech(sentence_text)
                        if clean_sentence and len(clean_sentence) > 1:
                            yield clean_sentence

                    sentence_buffer = parts[-1]

            print()

            leftover = self._clean_text_for_speech(sentence_buffer)
            if leftover and len(leftover) > 1:
                yield leftover

            clean_full = self._clean_text_for_speech(full_reply)
            self.conversation_history.append({"role": "assistant", "content": clean_full})
            return clean_full

        except Exception as e:
            print(f"\n[Brain Error]: {e}")
            if "10061" in str(e) or "Connection refused" in str(e) or "ConnectError" in str(e):
                print("[Brain Tip]: Ollama server disconnected or offline. Run 'ollama serve' in a terminal.")
            fallback = "Sorry machi, my mind wandered for a second. What were you saying?"
            self.conversation_history.append({"role": "assistant", "content": fallback})
            yield fallback
            return fallback

    def chat(self, user_input: str, memory_context: str = "") -> str:
        """Sends user input and returns the complete reply (for non-streaming calls)."""
        sentences = list(self.stream_sentences(user_input, memory_context))
        return " ".join(sentences)

if __name__ == "__main__":
    print("--- Jarvis Brain Initialized (Type 'exit' to quit) ---")
    brain = Brain()

    while True:
        try:
            user_msg = input("\nYou: ")
            if user_msg.lower() in ["exit", "quit", "bye"]:
                print("Jarvis: Catch you later bro. Take care.")
                break

            if not user_msg.strip():
                continue

            print("Jarvis: ", end="", flush=True)
            brain.chat(user_msg)

        except KeyboardInterrupt:
            print("\nExiting...")
            break