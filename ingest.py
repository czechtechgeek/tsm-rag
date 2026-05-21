#!/usr/bin/env python3
"""
TSM-RAG Ingest — load logs, create embeddings, store in ChromaDB.

Usage:
    python ingest.py                          # default: all-MiniLM-L6-v2 + chroma_db/
    python ingest.py --backend onnx           # ONNX Runtime backend
    python ingest.py --data custom_logs.json  # custom data file
    python ingest.py --reindex                # force re-index all logs

Example:
    python ingest.py
    python ingest.py --reindex
    python ingest.py --backend onnx --data ../other_logs.json
"""

import json
import logging
import os
import sys
import time

import click

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.src.embedder import create_embedder
from rag.src.chroma_client import ChromaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_FILE = os.path.join("sample_data", "logs.json")
DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_BACKEND = "sentence_transformers"


def load_logs(data_path: str):
    """Load log entries from JSON file."""
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.error(f"Run from project root (tsm-rag/) or provide --data path")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    logger.info(f"Loaded {len(logs)} log entries from {data_path}")
    return logs


def check_collection_stats(client: ChromaClient):
    """Print current collection stats."""
    count = client.count()
    logger.info(f"ChromaDB has {count} records")

    if count > 0:
        # Sample a few entries
        logger.info("✓ Database already populated — use query.py to search")
    else:
        logger.info("Database is empty — run 'python ingest.py' first")


@click.command()
@click.option(
    "--data",
    default=DEFAULT_DATA_FILE,
    show_default=True,
    help="Path to JSON file with log entries",
)
@click.option(
    "--chroma-dir",
    default=DEFAULT_CHROMA_DIR,
    show_default=True,
    help="ChromaDB persistence directory",
)
@click.option(
    "--backend",
    default=DEFAULT_BACKEND,
    show_default=True,
    type=click.Choice(["sentence_transformers", "onnx", "hailo"]),
    help="Embedding backend",
)
@click.option(
    "--reindex",
    is_flag=True,
    default=False,
    help="Force re-index all logs (drop existing collection)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Load logs only, don't create embeddings or DB",
)
@click.option(
    "--stats-only",
    is_flag=True,
    default=False,
    help="Show ChromaDB stats without ingesting",
)
def main(data, chroma_dir, backend, reindex, dry_run, stats_only):
    """Ingest test logs into ChromaDB with embeddings."""

    logs = load_logs(data)

    if stats_only:
        client = ChromaClient(persist_directory=chroma_dir)
        client.get_or_create_collection()
        check_collection_stats(client)
        return

    # Connect to ChromaDB
    client = ChromaClient(persist_directory=chroma_dir)

    if reindex:
        logger.warning("Re-index requested — resetting collection")
        client._connect()
        try:
            client._client.delete_collection("error_logs")
            logger.info("Deleted existing collection 'error_logs'")
        except Exception:
            pass

    # Create embedder
    logger.info(f"Using embedding backend: {backend}")
    embedder = create_embedder(backend=backend)

    if dry_run:
        # Test one embedding to verify model loads
        logger.info("Dry run — testing single embedding...")
        t0 = time.time()
        test_embedding = embedder.encode(["Test: Modbus timeout"])
        dt = time.time() - t0
        logger.info(f"✓ Embedding OK ({len(test_embedding[0])} dims, {dt:.2f}s)")
        logger.info("Dry run complete. Run without --dry-run to ingest.")
        return

    # Create embeddings in batches (avoid memory issues with many logs)
    BATCH_SIZE = 32
    total_added = 0
    total_logs = len(logs)

    logger.info(f"Creating embeddings for {total_logs} logs in batches of {BATCH_SIZE}...")
    t_start = time.time()

    for i in range(0, total_logs, BATCH_SIZE):
        batch = logs[i : i + BATCH_SIZE]
        batch_texts = [
            log.get("message", "") + " " + log.get("resolution", "")
            for log in batch
        ]

        t0 = time.time()
        embeddings = embedder.encode(batch_texts)
        batch_time = time.time() - t0

        added = client.add_logs(batch, embeddings)
        total_added += added

        progress = min(i + BATCH_SIZE, total_logs)
        logger.info(
            f"  Batch {progress}/{total_logs} "
            f"({added} new, {batch_time:.2f}s)"
        )

    t_total = time.time() - t_start
    collection_count = client.count()

    logger.info("=" * 50)
    logger.info(f"✅ Ingest complete:")
    logger.info(f"   Total logs processed: {total_logs}")
    logger.info(f"   New vectors added:    {total_added}")
    logger.info(f"   Total in ChromaDB:    {collection_count}")
    logger.info(f"   Duplicates skipped:   {total_logs - total_added}")
    logger.info(f"   Time:                 {t_total:.1f}s")
    logger.info(f"   Embedding backend:    {backend}")
    logger.info(f"   ChromaDB dir:         {os.path.abspath(chroma_dir)}")
    logger.info("=" * 50)
    logger.info(f"Next: python query.py \"popis chyby\"")


if __name__ == "__main__":
    main()