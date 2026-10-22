import os
import sys
import re

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
    except ModuleNotFoundError:
        print("\n" + "=" * 65)
        print("[Environment Error] 'torch' is not installed in the active environment!")
        print(f"Current Python: {sys.executable}")
        print("\nYour project dependencies are installed in the Conda 'assistant-voice' environment.")
        print("To fix this, run these commands in your PowerShell terminal:")
        print("   deactivate")
        print("   conda activate assistant-voice")
        print("   python main.py")
        print("=" * 65 + "\n")
        sys.exit(1)
    except Exception:
        pass

import config
from ear import Ear
from brain import Brain
from mouth import Mouth
from memory import Memory

def extract_and_store_facts(user_speech: str, memory: Memory):
    """Detects self-identifying statements and persists them into the SQLite facts table."""
    lower = user_speech.lower().strip()
    
    # 1. Name detection
    name_match = re.search(r'(?:my name is|call me|i am)\s+([a-zA-Z]+)', lower)
    if name_match:
        name = name_match.group(1).capitalize()
        if name not in ["tired", "fine", "good", "okay", "sad", "happy", "bro", "machi"]:
            memory.set_fact("user_name", name)
            print(f"[Learned Fact] user_name = {name}")

    # 2. Occupation detection
    job_match = re.search(r'(?:i work as a?n?|my job is)\s+([a-zA-Z\s]+)', lower)
    if job_match:
        job = job_match.group(1).strip()
        memory.set_fact("occupation", job)
        print(f"[Learned Fact] occupation = {job}")

def extract_and_store_emotions(user_speech: str, memory: Memory):
    """Detects emotional, life event, or mood-related statements and stores them in ChromaDB."""
    lower = user_speech.lower()
    
    # Meaningful emotional indicators (ignoring generic greeting words like 'bro' or 'machi')
    emotional_triggers = [
        "feel", "feeling", "tired", "exhausted", "stressed", "stress",
        "tension", "anxious", "sad", "depressed", "lonely", "happy",
        "excited", "interview", "manager yelled", "fired", "promoted",
        "passed", "failed", "breakup", "headache", "sick", "not well"
    ]
    
    # Ensure it is a substantive thought, not a 1-word query
    if any(trigger in lower for trigger in emotional_triggers) and len(user_speech.split()) >= 3:
        memory.add_emotional_memory(f"User mentioned: {user_speech}")

def run_jarvis():
    print("=" * 60)
    print("[Jarvis] Initializing Jarvis Companion on ROG G15...")
    print("=" * 60)

    # 1. Initialize all 4 pillars
    memory = Memory()
    ear = Ear()
    mouth = Mouth()
    brain = Brain()

    print("\n[Jarvis] Online and ready!")
    print("-> Press ENTER to speak. Type 'exit' or 'bye' to quit.\n")

    while True:
        try:
            control = input("\n[Press ENTER to talk to Jarvis | Type 'exit' to quit]: ")
            if control.strip().lower() in ["exit", "quit", "bye"]:
                goodbye_msg = "Catch you later machi. Take care of yourself."
                print(f"\nJarvis: {goodbye_msg}")
                mouth.speak(goodbye_msg)
                break

            # --- 1. THE EAR: Listen to User ---
            user_speech = ear.listen_and_transcribe(duration=config.LISTEN_DURATION)
            if not user_speech or len(user_speech.strip()) < 2:
                print("[Warning] Didn't catch that clearly, try again.")
                continue

            # --- 2. THE MEMORY: Recall relevant facts & past memories ---
            facts = memory.get_all_facts()
            past_memories = memory.recall_memories(user_speech, limit=2)

            context_snippets = []
            if facts:
                facts_str = ", ".join([f"{k}: {v}" for k, v in facts.items()])
                context_snippets.append(f"Static profile: {facts_str}")
            if past_memories:
                context_snippets.append(f"Related past memories: {'; '.join(past_memories)}")
                print(f"[Recalled Context]: {past_memories}")

            memory_context = " | ".join(context_snippets) if context_snippets else ""

            # --- 3. THE BRAIN & MOUTH: Stream Sentences directly to Speech Pipeline ---
            print("\nJarvis: ", end="", flush=True)
            sentence_stream = brain.stream_sentences(user_speech, memory_context=memory_context)
            mouth.speak_stream(sentence_stream)

            # --- 4. THE MEMORY: Save important facts & emotions asynchronously ---
            extract_and_store_facts(user_speech, memory)
            extract_and_store_emotions(user_speech, memory)

        except KeyboardInterrupt:
            print("\nShutting down Jarvis...")
            break
        except Exception as e:
            print(f"\n[Warning] Loop error: {e}")

if __name__ == "__main__":
    run_jarvis()