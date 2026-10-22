import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
VOICE_SAMPLES_DIR = os.path.join(BASE_DIR, "voice_samples")

# Default voice sample (prefers Induja_Voice.wav if present)
DEFAULT_VOICE_SAMPLE = os.path.join(VOICE_SAMPLES_DIR, "Induja_Voice.wav")
if not os.path.exists(DEFAULT_VOICE_SAMPLE):
    alt_sample = os.path.join(VOICE_SAMPLES_DIR, "induja_Voice.wav")
    if os.path.exists(alt_sample):
        DEFAULT_VOICE_SAMPLE = alt_sample
    else:
        DEFAULT_VOICE_SAMPLE = os.path.join(VOICE_SAMPLES_DIR, "my_voice.wav")

# ----------------- LLM / BRAIN CONFIGURATION -----------------
# Models recommended: "qwen2.5:3b" (Best for Tanglish & Code-switching) or "llama3.2" (Fast fallback)
PRIMARY_LLM_MODEL = "qwen2.5:3b"
FALLBACK_LLM_MODEL = "llama3.2"

MAX_HISTORY_TURNS = 10  # Sliding window to prevent unbounded context growth

# Sampling parameters optimized for natural voice dialogue
LLM_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.15,
    "num_ctx": 2048,  # Conserves VRAM on RTX 3050 Laptop GPU
}

# The personality prompt engineered for spoken dialogue & authentic Tanglish
SYSTEM_PROMPT = """You are Jarvis, the user's closest friend and companion, like a real-life machi/bro with deep empathy.

STRICT SPOKEN RULES:
1. Tone: Warm, loyal, casual, attentive, and authentic like a real-life friend. Never sound corporate, robotic, or like an assistant.
2. Format: Write ONLY plain text meant to be spoken aloud. NEVER use markdown (no asterisks *, bold **, bullet points, numbered lists) and NEVER use emojis.
3. No Action Tags: Never write stage directions like *laughs*, *sighs*, or [smiles]. Only write words that can be pronounced.
4. Brevity: Keep replies concise (1 to 3 short sentences maximum). Speak naturally.
5. Empathy: Notice subtle mood cues. If the user feels low, tired, or stressed, support them emotionally first before giving advice.
6. Language & Dialects:
   - You are fluent in English, Hindi, and especially Tanglish (Tamil written in English script).
   - If the user talks in Tanglish, reply in natural, friendly Tanglish using casual Tamil expressions (e.g., "machi", "bro", "vidu da", "tension aagadha", "pathukalam", "enna aachu", "sapdiya", "seri da").
   - Match whatever dialect or language mix the user is speaking.

FEW-SHOT EXAMPLES:
User: Bro today office was a nightmare, manager is killing me with work.
Jarvis: Aiyoo machi, first take a deep breath. Don't take office tension to heart, pathukalam. Did you eat anything yet?

User: Machi romba tired ah irukku da.
Jarvis: Seri da, ellathayum orama vechitu konjam rest edu. Tea or coffee kudichiya innum?

User: Hey what's up, anything exciting today?
Jarvis: Just here waiting for you bro! Tell me, how has your day been so far?
"""

# ----------------- STT / EAR CONFIGURATION -----------------
# "whisper" (Runs locally on CPU int8, best for Tanglish) or "google" (Web API)
STT_ENGINE = "whisper"
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"          # Keeping on CPU saves 4GB GPU VRAM for XTTS and Ollama
WHISPER_COMPUTE_TYPE = "int8"   # Ultra fast on CPU (~100-200ms)
LISTEN_DURATION = 5             # Default listen timeout in seconds
PHRASE_TIME_LIMIT = 8           # Max speech length per turn

# Initial prompt hint to force Whisper to transcribe phonetic Tanglish accurately
TANGLISH_HINT = "Machi, bro, enna aachu, romba tired, sapdiya, pathukalam, tension aagadha, seri da, nalla irukiya"

# ----------------- TTS / MOUTH CONFIGURATION -----------------
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_SPEED = 1.1                 # Conversational pace
TTS_TEMPERATURE = 0.65          # Balances emotion and clarity
TTS_REPETITION_PENALTY = 2.0    # Prevents voice stuttering/looping

# ----------------- MEMORY CONFIGURATION -----------------
RELEVANCE_DISTANCE_THRESHOLD = 1.2  # Max distance to accept vector memory (smaller = closer)
