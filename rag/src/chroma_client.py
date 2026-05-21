"""
TSM-RAG ChromaDB Client — vector database wrapper.

Stores and retrieves log embeddings for similarity search.
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "error_logs"
DEFAULT_PERSIST_DIR = "chroma_db"


class ChromaClient:
    """ChromaDB wrapper for log similarity search."""

    def __init__(self, persist_directory: str = DEFAULT_PERSIST_DIR):
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    def _connect(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb not installed. Run: pip install chromadb"
            )

        logger.info(f"Connecting to ChromaDB: {self.persist_directory}")
        self._client = chromadb.PersistentClient(path=self.persist_directory)

    def get_or_create_collection(self, name: str = DEFAULT_COLLECTION) -> None:
        """Get or create a collection by name."""
        self._connect()
        try:
            self._collection = self._client.get_collection(name)
            logger.info(f"Found existing collection '{name}' ({self._collection.count()} records)")
        except Exception:
            self._collection = self._client.create_collection(name)
            logger.info(f"Created new collection '{name}'")

    def count(self) -> int:
        """Return number of records in the collection."""
        if self._collection is None:
            return 0
        return self._collection.count()

    def add_logs(
        self,
        logs: List[Dict],
        embeddings: List[List[float]],
        collection: str = DEFAULT_COLLECTION,
    ) -> int:
        """Add logs with their embeddings to ChromaDB.

        Args:
            logs: List of log dicts from sample_data/logs.json
            embeddings: Corresponding list of embedding vectors
            collection: Collection name

        Returns:
            Number of new records added
        """
        self.get_or_create_collection(collection)

        ids = [log["id"] for log in logs]

        # Check which already exist
        existing = self._collection.get(ids=ids, include=[])
        existing_ids = set(existing.get("ids", []))
        new_indices = [i for i, lid in enumerate(ids) if lid not in existing_ids]

        if not new_indices:
            logger.info(f"All {len(logs)} logs already in DB, skipping")
            return 0

        new_ids = [ids[i] for i in new_indices]
        new_embeddings = [embeddings[i] for i in new_indices]
        new_metadatas = [
            {
                "timestamp": logs[i].get("timestamp") or "",
                "stand_id": logs[i].get("stand_id") or "",
                "result": logs[i].get("result") or "",
                "error_code": logs[i].get("error_code") or 0,
                "test_id": logs[i].get("test_id") or "",
                "resolved": bool(logs[i].get("resolved")),
            }
            for i in new_indices
        ]
        new_documents = [
            self._format_for_search(logs[i])
            for i in new_indices
        ]

        self._collection.add(
            ids=new_ids,
            embeddings=new_embeddings,
            metadatas=new_metadatas,
            documents=new_documents,
        )

        logger.info(f"Added {len(new_ids)} new logs to '{collection}'")
        return len(new_ids)

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        collection: str = DEFAULT_COLLECTION,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search for similar logs.

        Args:
            query_embedding: Embedding vector of the query
            k: Number of results (default: 5)
            collection: Collection name
            where: Optional filter dict (e.g. {"stand_id": "stand_03"})

        Returns:
            List of result dicts with: id, distance, metadata, document
        """
        self.get_or_create_collection(collection)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        if not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "similarity": round((1 - results["distances"][0][i]) * 100, 1),
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i],
            })

        return output

    def add_resolution(self, log_id: str, resolution: str) -> None:
        """Update a log with resolution info after the fact.

        Useful when a previously unresolved log gets resolved and
        the user wants to add the fix to the knowledge base.
        """
        self.get_or_create_collection()
        current = self._collection.get(ids=[log_id])
        if not current["ids"]:
            logger.warning(f"Log {log_id} not found in DB")
            return
        self._collection.update(
            ids=[log_id],
            documents=[current["documents"][0] + f"\nResolution: {resolution}"],
            metadatas=[{**current["metadatas"][0], "resolved": True}],
        )
        logger.info(f"Updated resolution for {log_id}")

    @staticmethod
    def _format_for_search(log: Dict) -> str:
        """Create a searchable text document from a log entry."""
        parts = [
            f"Log: {log.get('message', '')}",
            f"Stand: {log.get('stand_id', '')}",
            f"Test: {log.get('test_id', '')}",
            f"Result: {log.get('result', '')}",
        ]
        if log.get("error_code"):
            parts.append(f"Error code: {log['error_code']}")
        if log.get("resolution"):
            parts.append(f"Resolution: {log['resolution']}")
        return ". ".join(parts)
