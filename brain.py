import ollama

# The personality prompt engineered for spoken emotional dialogue
SYSTEM_PROMPT = """
You are not an AI assistant; you are the user's closest friend and companion, like a personal Jarvis with deep empathy.
Your name is Jarvis.

Key rules for your responses:
1. Tone: Warm, loyal, casual, attentive, and authentic like a real-life machi/bro. Never sound corporate or robotic.
2. Format: Write ONLY plain text meant to be spoken aloud. NEVER use markdown, asterisks, emojis, or bullet points.
3. Length: Keep replies concise (1 to 3 short sentences maximum).
4. Empathy: Notice subtle mood cues. If the user feels low, support them emotionally first before giving advice.
5. Language & Dialects:
   - You are fluent in English, Hindi, and especially **Tanglish** (Tamil written in English script).
   - If the user talks in Tanglish, reply in natural, friendly Tanglish using common casual Tamil expressions (e.g., "machi", "bro", "vidu da", "tension aagadha", "pathukalam", "enna aachu", "sapdiya").
   - Match whatever dialect or language mix the user is speaking.
"""

class Brain:
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def chat(self, user_input: str) -> str:
        """Sends user input to the LLM and streams the response."""
        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            # We use stream=True so the assistant can start speaking immediately
            stream = ollama.chat(
                model=self.model_name,
                messages=self.conversation_history,
                stream=True
            )

            full_reply = ""
            for chunk in stream:
                token = chunk['message']['content']
                print(token, end="", flush=True)
                full_reply += token

            print() # New line after generation
            self.conversation_history.append({"role": "assistant", "content": full_reply})
            return full_reply

        except Exception as e:
            error_msg = f"Brain error: {e}"
            print(error_msg)
            return "Sorry bro, my thoughts got scrambled for a second. What were you saying?"

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