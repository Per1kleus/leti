"""
Long-term memory: stores facts, preferences, and past exchanges as embeddings
in ChromaDB for semantic recall across sessions. Uses Ollama's embedding
endpoint so everything stays local.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

import chromadb

from core.config_loader import get_settings, resolve_path
from core.llm_client import OllamaClient


class VectorMemory:
    @property
    def settings(self) -> Dict[str, Any]:
        """Read live so /settings edits apply without a restart (persist_dir is bound to the open Chroma client and still needs one)."""
        return get_settings()["memory"]

    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
        persist_dir = str(resolve_path(self.settings["persist_dir"]))
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(name="leti_long_term_memory")

    async def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Embed and store a fact/preference/summary for later semantic recall."""
        embedding = await self.llm_client.embed(text)
        memory_id = str(uuid.uuid4())
        meta = {"created_at": time.time(), **(metadata or {})}
        self._collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )
        return memory_id

    async def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Semantic search over long-term memory, returning best-matching facts."""
        k = top_k or self.settings.get("long_term_top_k", 5)
        embedding = await self.llm_client.embed(query)
        results = self._collection.query(query_embeddings=[embedding], n_results=k)

        matches = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, distances):
            matches.append({"text": doc, "metadata": meta, "distance": dist})
        return matches

    def delete(self, memory_id: str) -> None:
        self._collection.delete(ids=[memory_id])

    def count(self) -> int:
        return self._collection.count()
