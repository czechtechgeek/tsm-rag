#!/usr/bin/env python3
"""
TSM-RAG Query — CLI tool for searching similar logs.

Usage:
    python query.py "Modbus timeout slave 48"
    python query.py --top-k 10 "CRC error stand 01"
    python query.py --stand stand_03 "voltage issue"
    python query.py --web                      # start Flask UI
    python query.py --stats                    # show DB stats
    python query.py --interactive              # interactive REPL mode
"""

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


HEADER = """
╔══════════════════════════════════════════════╗
║  TSM-RAG — Test Stand Monitor Query Engine  ║
╚══════════════════════════════════════════════╝
"""


def format_results(results: list, query: str) -> str:
    """Format search results for terminal output."""
    if not results:
        return "\n⚠️  No similar logs found. Try a different query.\n"

    lines = [
        HEADER,
        f"🔍 Hledám podobné: \"{query}\"",
        "",
        f"Výsledky ({len(results)} nejpodobnějších):",
        "─" * 60,
    ]

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        doc = r["document"]
        similarity = r["similarity"]

        # Extract first line of document (the message)
        msg = doc.split(". ")[0] if doc else ""
        msg = msg.replace("Log: ", "")

        # Resolution status
        resolved = meta.get("resolved", False)
        status_icon = "✓" if resolved else "✗"
        status_text = "Vyřešeno" if resolved else "Nevyřešeno"

        lines.append(f"\n{i}. {r['id']} | {meta.get('stand_id', '?')} | {meta.get('timestamp', '?')}")
        lines.append(f"   \"{msg}\"")
        lines.append(f"   Podobnost: {similarity}%")
        lines.append(f"   {status_icon} {status_text}")

        # Extract resolution from document if available
        if "Resolution:" in doc:
            resolution = doc.split("Resolution:")[-1].strip()
            lines.append(f"   Řešení: {resolution}")

    lines.append("\n" + "─" * 60)
    return "\n".join(lines)


@click.group(invoke_without_command=True)
@click.option(
    "--chroma-dir",
    default="chroma_db",
    show_default=True,
    help="ChromaDB persistence directory",
)
@click.option(
    "--backend",
    default="sentence_transformers",
    show_default=True,
    help="Embedding backend",
)
@click.option("--top-k", default=5, show_default=True, help="Number of results")
@click.option("--stand", default=None, help="Filter by stand ID (e.g. stand_03)")
@click.option("--web", is_flag=True, default=False, help="Start Flask web UI")
@click.option("--stats", is_flag=True, default=False, help="Show DB statistics")
@click.option("--interactive", is_flag=True, default=False, help="Interactive mode")
@click.argument("query", required=False)
@click.pass_context
def cli(ctx, chroma_dir, backend, top_k, stand, web, stats, interactive, query):
    """Search for similar test log entries."""

    ctx.ensure_object(dict)
    ctx.obj["chroma_dir"] = chroma_dir
    ctx.obj["backend"] = backend
    ctx.obj["top_k"] = top_k
    ctx.obj["stand"] = stand

    if stats:
        client = ChromaClient(persist_directory=chroma_dir)
        client.get_or_create_collection()
        count = client.count()

        print(HEADER)
        print(f"📊 ChromaDB Statistics:")
        print(f"   Collection:    error_logs")
        print(f"   Records:       {count}")
        print(f"   Persist dir:   {os.path.abspath(chroma_dir)}")
        if count == 0:
            print("\n⚠️  Database is empty! Run 'python ingest.py' first.")
        return

    if web:
        _start_webui(chroma_dir, backend)
        return

    if not query and not interactive:
        click.echo(cli.get_help(ctx))
        return

    if interactive:
        _interactive_mode(chroma_dir, backend, top_k, stand)
        return

    # Single query mode
    _perform_search(query, chroma_dir, backend, top_k, stand)


@cli.command()
@click.pass_context
def web(ctx):
    """Start the Flask web UI."""
    obj = ctx.obj
    _start_webui(obj["chroma_dir"], obj["backend"])


def _perform_search(query, chroma_dir, backend, top_k, stand):
    """Execute a single search query."""
    print(HEADER)

    # Check ChromaDB exists
    client = ChromaClient(persist_directory=chroma_dir)
    client.get_or_create_collection()

    if client.count() == 0:
        print("⚠️  Database is empty! Run 'python ingest.py' first.")
        return

    # Embed the query
    t0 = time.time()
    embedder = create_embedder(backend=backend)
    query_embedding = embedder.encode([query])[0]
    embed_time = time.time() - t0

    # Search
    where = {"stand_id": stand} if stand else None
    t0 = time.time()
    results = client.search(query_embedding, k=top_k, where=where)
    search_time = time.time() - t0

    # Output
    output = format_results(results, query)
    output += f"\n⏱  Embedding: {embed_time:.2f}s | Search: {search_time:.2f}s | Backend: {backend}"
    print(output)


def _interactive_mode(chroma_dir, backend, top_k, stand):
    """Interactive REPL for repeated queries."""
    embedder = create_embedder(backend=backend)
    client = ChromaClient(persist_directory=chroma_dir)
    client.get_or_create_collection()

    print(HEADER)
    print("💬 Interactive mode — enter queries or 'exit'/'quit' to stop.")
    print(f"   ChromaDB: {client.count()} records | Backend: {backend} | Top-K: {top_k}")
    if stand:
        print(f"   Filtered to stand: {stand}")
    print()

    while True:
        try:
            query = input("🔍 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            break

        # Commands
        if query.startswith(":"):
            cmd = query[1:].strip().lower()
            if cmd.startswith("topk "):
                try:
                    top_k = int(cmd.split()[1])
                    print(f"   → Top-K now: {top_k}")
                except (IndexError, ValueError):
                    print("   Usage: :topk 10")
                continue
            elif cmd.startswith("stand "):
                stand = cmd.split()[1] if len(cmd.split()) > 1 else None
                if stand == "all" or stand == "*":
                    stand = None
                    print("   → Stand filter: ALL")
                else:
                    print(f"   → Stand filter: {stand}")
                continue
            elif cmd == "stats":
                print(f"   Collection: {client.count()} records")
                continue
            else:
                print(f"   Unknown command: {cmd}")
                continue

        # Normal query
        query_emb = embedder.encode([query])[0]
        results = client.search(query_emb, k=top_k, where={"stand_id": stand} if stand else None)
        print(format_results(results, query))
        print()


def _start_webui(chroma_dir, backend):
    """Start Flask web application."""
    print(f"🚀 Starting web UI on http://127.0.0.1:5000")
    print(f"   Backend: {backend}  |  ChromaDB: {chroma_dir}")
    print("   Press Ctrl+C to stop.")
    print()

    # Import and run app
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag", "src"))
    from app import create_app

    app = create_app(chroma_dir=chroma_dir, backend=backend)
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    cli()