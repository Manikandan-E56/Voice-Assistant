import os
import sqlite3
import chromadb
from datetime import datetime

class Memory:
    def __init__(self, db_folder="memory"):
        os.makedirs(db_folder, exist_ok=True)
        self.sqlite_path = os.path.join(db_folder, "facts.db")
        self.chroma_path = os.path.join(db_folder, "chroma_store")
        
        # 1. Initialize SQLite for Key Facts
        self._init_sqlite()
        
        # 2. Initialize ChromaDB for Emotional/Semantic Memories
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(name="user_memories")
        print("Memory Bank initialized successfully!")

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
            (key.lower(), value, now)
        )
        conn.commit()
        conn.close()

    def get_fact(self, key: str) -> str:
        """Fetch a specific fact."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM facts WHERE key = ?", (key.lower(),))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    # --- Semantic / Emotional Memory Methods (ChromaDB) ---
    def add_emotional_memory(self, memory_text: str):
        """Saves an event, feeling, or conversation highlight into vector memory."""
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        self.collection.add(
            documents=[memory_text],
            metadatas=[{"timestamp": timestamp}],
            ids=[memory_id]
        )
        print(f"Stored Memory: \"{memory_text}\"")

    def recall_memories(self, query: str, limit: int = 2) -> list:
        """Searches past memories relevant to what the user is currently talking about."""
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(limit, count)
        )
        return results['documents'][0] if results['documents'] else []

    def get_memory_count(self):
        return self.collection.count()