"""
TSM-RAG Flask Web UI — search interface for test log similarity.

Run via:
    python query.py --web
    # or
    python -m rag.src.app
"""

import os
import sys
import time

from flask import Flask, render_template_string, request

# Adjust path for when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rag.src.embedder import create_embedder
from rag.src.chroma_client import ChromaClient

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="cs">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TSM-RAG — Test Stand Monitor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
        }
        .container { max-width: 960px; margin: 0 auto; padding: 2rem 1rem; }
        h1 {
            font-size: 1.5rem;
            color: #38bdf8;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .subtitle { color: #64748b; margin-bottom: 1.5rem; font-size: 0.9rem; }
        .search-box {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }
        .search-box input {
            flex: 1;
            padding: 0.75rem 1rem;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            color: #e2e8f0;
            font-size: 1rem;
        }
        .search-box input:focus {
            outline: none;
            border-color: #38bdf8;
        }
        .search-box button {
            padding: 0.75rem 1.5rem;
            background: #38bdf8;
            color: #0f172a;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            font-size: 1rem;
        }
        .search-box button:hover { background: #7dd3fc; }
        .filters {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }
        .filters select, .filters input {
            padding: 0.5rem;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 6px;
            color: #e2e8f0;
            font-size: 0.875rem;
        }
        .stats {
            background: #1e293b;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 1.5rem;
            display: flex;
            gap: 1.5rem;
            font-size: 0.875rem;
            color: #94a3b8;
        }
        .stats strong { color: #e2e8f0; }
        .result-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            transition: border-color 0.2s;
        }
        .result-card:hover { border-color: #475569; }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        .result-id { font-family: monospace; font-size: 0.8rem; color: #64748b; }
        .result-stand {
            font-size: 0.8rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background: #0f172a;
        }
        .result-stand.stand_01 { color: #f87171; }
        .result-stand.stand_02 { color: #fb923c; }
        .result-stand.stand_03 { color: #fbbf24; }
        .result-stand.stand_04 { color: #4ade80; }
        .result-stand.stand_05 { color: #38bdf8; }
        .result-stand.stand_06 { color: #a78bfa; }
        .result-message {
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }
        .result-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.8rem;
            color: #64748b;
            flex-wrap: wrap;
        }
        .result-meta span { display: flex; align-items: center; gap: 0.25rem; }
        .similarity-bar {
            display: inline-block;
            height: 6px;
            border-radius: 3px;
            background: #0f172a;
            width: 80px;
            vertical-align: middle;
            margin-right: 0.5rem;
        }
        .similarity-fill {
            display: block;
            height: 100%;
            border-radius: 3px;
            background: linear-gradient(90deg, #22c55e, #38bdf8);
        }
        .resolution {
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: #0f172a;
            border-radius: 6px;
            font-size: 0.85rem;
            color: #86efac;
        }
        .no-results {
            text-align: center;
            padding: 3rem;
            color: #64748b;
        }
        .no-results .icon { font-size: 3rem; margin-bottom: 1rem; }
        .footer {
            text-align: center;
            padding: 2rem;
            color: #334155;
            font-size: 0.8rem;
        }
        .pass { color: #22c55e; }
        .fail { color: #ef4444; }
        .error { color: #f59e0b; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 TSM-RAG</h1>
        <div class="subtitle">Test Stand Monitor — vyhledávání v historii chyb</div>

        <form method="POST">
            <div class="search-box">
                <input type="text" name="query" placeholder="Popiš chybu... např. 'Modbus timeout na slave 48'"
                       value="{{ query }}" autofocus>
                <button type="submit">Hledat</button>
            </div>
            <div class="filters">
                <select name="stand">
                    <option value="">Všechny standy</option>
                    {% for s in stands %}
                    <option value="{{ s }}" {% if stand == s %}selected{% endif %}>{{ s }}</option>
                    {% endfor %}
                </select>
                <select name="top_k">
                    {% for k in [5, 10, 15, 20] %}
                    <option value="{{ k }}" {% if top_k == k %}selected{% endif %}>{{ k }} výsledků</option>
                    {% endfor %}
                </select>
            </div>
        </form>

        {% if stats %}
        <div class="stats">
            <span>📊 <strong>{{ stats.count }}</strong> záznamů v DB</span>
            <span>⚡ <strong>{{ "%.2fs"|format(stats.time) }}</strong> odezva</span>
            <span>🧠 <strong>{{ stats.backend }}</strong></span>
        </div>
        {% endif %}

        {% if results %}
            {% for r in results %}
            <div class="result-card">
                <div class="result-header">
                    <span class="result-id">{{ r.id }}</span>
                    <span class="result-stand {{ r.metadata.stand_id }}">{{ r.metadata.stand_id }}</span>
                </div>
                <div class="result-message">{{ r.document.split('. ')[0].replace('Log: ', '') }}</div>
                <div class="result-meta">
                    <span>📅 {{ r.metadata.timestamp }}</span>
                    <span>🎯 Podobnost:
                        <span class="similarity-bar">
                            <span class="similarity-fill" style="width: {{ r.similarity }}%"></span>
                        </span>
                        <strong>{{ "%.1f"|format(r.similarity) }}%</strong>
                    </span>
                    <span class="{{ r.metadata.result|lower }}">{{ r.metadata.result }}</span>
                    <span>{% if r.metadata.resolved == True %}✅ Vyřešeno{% else %}❌ Nevyřešeno{% endif %}</span>
                </div>
                {% if 'Resolution:' in r.document %}
                <div class="resolution">💡 {{ r.document.split('Resolution:')[-1].strip() }}</div>
                {% endif %}
            </div>
            {% endfor %}
        {% elif query %}
            <div class="no-results">
                <div class="icon">🔍</div>
                <p>Žádné podobné logy nenalezeny. Zkus jiný popis chyby.</p>
            </div>
        {% else %}
            <div class="no-results">
                <div class="icon">🔬</div>
                <p>Zadej popis chyby výše a najdi podobné historické případy.</p>
                <p style="margin-top: 0.5rem; font-size: 0.85rem;">
                    Např.: <em>"Modbus timeout slave 48"</em>, <em>"CRC error stand 01"</em>, <em>"voltage out of range"</em>
                </p>
            </div>
        {% endif %}

        <div class="footer">
            TSM-RAG &middot; Embedding: {{ embedder_name }} &middot; ChromaDB perzistentní
        </div>
    </div>
</body>
</html>
"""


def create_app(chroma_dir: str = "chroma_db", backend: str = "sentence_transformers"):
    """Create and configure the Flask application."""
    app = Flask(__name__)

    embedder = create_embedder(backend=backend)
    client = ChromaClient(persist_directory=chroma_dir)
    client.get_or_create_collection()

    # Get list of stands from DB
    stands = set()
    all_data = client._collection.get(include=["metadatas"])
    if all_data and all_data.get("metadatas"):
        for meta in all_data["metadatas"]:
            if "stand_id" in meta:
                stands.add(meta["stand_id"])

    @app.route("/", methods=["GET", "POST"])
    def index():
        query = request.form.get("query", "").strip()
        stand = request.form.get("stand", "")
        top_k = int(request.form.get("top_k", 5))

        results = []
        stats = None

        if query:
            # Prevent empty embedding on single space
            if not query.strip():
                return render_template_string(
                    HTML_TEMPLATE, query="", stand=stand,
                    top_k=top_k, stands=sorted(stands),
                    results=None, stats=None, embedder_name=backend
                )

            t0 = time.time()
            query_embedding = embedder.encode([query])[0]
            where = {"stand_id": stand} if stand else None
            results = client.search(query_embedding, k=top_k, where=where)
            dt = time.time() - t0

            stats = {
                "count": client.count(),
                "time": dt,
                "backend": backend,
            }

        return render_template_string(
            HTML_TEMPLATE,
            query=query,
            stand=stand,
            top_k=top_k,
            stands=sorted(stands),
            results=results,
            stats=stats,
            embedder_name=backend,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    print("🚀 TSM-RAG Web UI on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)