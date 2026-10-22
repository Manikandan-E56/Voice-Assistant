import os
import sys
import sqlite3
import chromadb
from datetime import datetime
from typing import List, Dict, Optional
import config

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

class Memory:
    def __init__(self, db_folder: str = None):
        self.db_folder = db_folder or config.MEMORY_DIR
        os.makedirs(self.db_folder, exist_ok=True)
        self.sqlite_path = os.path.join(self.db_folder, "facts.db")
        self.chroma_path = os.path.join(self.db_folder, "chroma_store")

        # 1. Initialize SQLite for Key Facts
        self._init_sqlite()

        # 2. Initialize ChromaDB for Emotional/Semantic Memories
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(name="user_memories")
        print(f"[Memory Bank] Initialized (Stored memories: {self.collection.count()})")

    def _init_sqlite(self):
        """Creates the key-value facts table if it doesn't exist."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    # --- Facts Methods (SQLite) ---
    def set_fact(self, key: str, value: str):
        """Save a static fact like name, job, favorite thing."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT OR REPLACE INTO facts (key, value, updated_at) VALUES (?, ?, ?)",
            (key.strip().lower(), value.strip(), now)
        )
        conn.commit()
        conn.close()

    def get_fact(self, key: str) -> Optional[str]:
        """Fetch a specific fact."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM facts WHERE key = ?", (key.strip().lower(),))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def get_all_facts(self) -> Dict[str, str]:
        """Returns all persistent static facts as a dictionary."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM facts")
        rows = cursor.fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}

    # --- Semantic / Emotional Memory Methods (ChromaDB) ---
    def add_emotional_memory(self, memory_text: str):
        """Saves an event, feeling, or conversation highlight into vector memory."""
        memory_text = memory_text.strip()
        if not memory_text or len(memory_text) < 5:
            return

        # Avoid storing exact duplicate if it already exists
        existing = self.recall_memories(memory_text, limit=1)
        if existing and existing[0].lower() == memory_text.lower():
            return

        memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        self.collection.add(
            documents=[memory_text],
            metadatas=[{"timestamp": timestamp}],
            ids=[memory_id]
        )
        print(f"[Memory Bank] Stored: \"{memory_text}\"")

    def recall_memories(self, query: str, limit: int = 2) -> List[str]:
        """
        Searches past memories relevant to what the user is currently talking about.
        Filters out low-relevance matches based on vector distance.
        """
        count = self.collection.count()
        if count == 0 or not query.strip():
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(limit, count),
                include=["documents", "distances"]
            )

            if not results or not results.get("documents") or not results["documents"][0]:
                return []

            docs = results["documents"][0]
            distances = results.get("distances", [[]])[0]

            relevant_memories = []
            for doc, dist in zip(docs, distances):
                # Only include memories that meet distance threshold
                if dist <= config.RELEVANCE_DISTANCE_THRESHOLD:
                    relevant_memories.append(doc)

            return relevant_memories

        except Exception as e:
            print(f"[Memory Error]: {e}")
            return []

    def get_memory_count(self) -> int:
        return self.collection.count()

if __name__ == "__main__":
    mem = Memory()
    mem.set_fact("user_name", "Boss")
    print("Facts:", mem.get_all_facts())
    mem.add_emotional_memory("User was feeling exhausted from work meetings today.")
    print("Recalled:", mem.recall_memories("I had so many meetings at office"))