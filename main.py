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

torch.backends.cudnn.enabled = False
from ear import Ear
from brain import Brain
from mouth import Mouth
from memory import Memory

def run_jarvis():
    print("=" * 55)
    print("🤖 INITIALIZING JARVIS COMPANION ON ROG G15...")
    print("=" * 55)

    # 1. Initialize all 4 pillars
    memory = Memory()
    ear = Ear()
    mouth = Mouth(voice_sample_path="voice_samples/induja_Voice.wav")
    brain = Brain(model_name="llama3.2")

    print("\n✅ Jarvis is fully online and ready!")
    print("👉 Press ENTER to speak. Say 'exit' or 'bye' to quit.\n")

    while True:
        try:
            # Wait for user prompt to start speaking (prevents picking up background noise)
            control = input("\n[Press ENTER to talk to Jarvis | Type 'exit' to quit]: ")
            if control.strip().lower() in ["exit", "quit", "bye"]:
                goodbye_msg = "Catch you later machi. Take care of yourself."
                print(f"\nJarvis: {goodbye_msg}")
                mouth.speak(goodbye_msg)
                break

            # --- 1. THE EAR: Listen to User ---
            user_speech = ear.listen_and_transcribe(duration=5)
            if not user_speech or len(user_speech.strip()) < 2:
                print("⚠️ Didn't catch that clearly, try again.")
                continue

            # --- 2. THE MEMORY: Recall relevant past context ---
            past_memories = memory.recall_memories(user_speech, limit=2)
            memory_context = ""
            if past_memories:
                memory_context = f"[Context from past memories you know about user: {'; '.join(past_memories)}]\n"
                print(f"🧠 Recalled Context: {past_memories}")

            # --- 3. THE BRAIN: Generate empathetic response ---
            augmented_prompt = f"{memory_context}User says: {user_speech}"
            print("Jarvis is thinking...")
            reply = brain.chat(augmented_prompt)

            # --- 4. THE VOICE: Speak response in cloned voice ---
            mouth.speak(reply)

            # --- 5. THE MEMORY: Save important notes asynchronously ---
            # If the user shared something emotional or personal, store it
            if any(word in user_speech.lower() for word in ["feel", "tired", "stressed", "sad", "happy", "office", "work", "manager", "interview", "love", "hate", "machi", "bro"]):
                memory.add_emotional_memory(f"User mentioned: {user_speech}")

        except KeyboardInterrupt:
            print("\nShutting down Jarvis...")
            break
        except Exception as e:
            print(f"\n⚠️ Loop error: {e}")

if __name__ == "__main__":
    run_jarvis()