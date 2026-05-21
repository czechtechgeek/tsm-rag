# TSM-RAG — Test Stand Monitor s RAG engine

**Cíl:** Centrální monitoring testovacích standů s AI diagnostikou na **Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H)**

## 🚀 Rychlý start

```bash
cd ~/tsm-rag

# 1. Instalace závislostí
pip install -r requirements.txt

# 2. Ingest testovacích logů do ChromaDB
python ingest.py

# 3. Hledání podobných chyb
python query.py "Modbus timeout slave 48"

# 4. Webové rozhraní
python query.py --web
# → http://localhost:5000
```

## 📁 Struktura projektu

```
~/tsm-rag/
├── ingest.py              # Načtení logů → embedding → ChromaDB
├── query.py               # CLI vyhledávání + Flask web UI
├── model_tester.py        # Testování malých LLM na RPi 5
├── requirements.txt
├── README.md
│
├── sample_data/
│   └── logs.json          # 61 testovacích logů (Jablotron/Wavin)
│
├── rag/
│   └── src/
│       ├── embedder.py       # Embedding backend (CPU/ONNX/Hailo)
│       ├── chroma_client.py  # ChromaDB wrapper
│       ├── hailo_runner.py   # Hailo-10H inference (prototype)
│       └── app.py            # Flask web UI
│
├── collector/             # (Phase 1 — sběr z reálných standů)
│   ├── config/
│   │   ├── stands.yaml   # Definice standů
│   │   └── rules.yaml    # Evaluační pravidla
│   └── src/
│       └── models.py     # SQLAlchemy schéma
│
└── models/
    └── hailo/            # .hef modely pro Hailo-10H
```

## 🧠 Malé modely k vyzkoušení

| Model | Parametry | RAM | Rychlost na CPU | Doporučení |
|-------|-----------|-----|-----------------|------------|
| **all-MiniLM-L6-v2** | 22 MB | 0.5 GB | ~800 vět/s | ✅ Embedding, vždy funguje |
| **SmolLM2-360M** | 360M | 1.5 GB | ~20 tok/s | 🔬 Nejmenší LLM test |
| **TinyLlama-1.1B** | 1.1B | 3 GB | ~8 tok/s | ⭐ Dobrý kompromis |
| **Qwen2.5-1.5B** | 1.5B | 4 GB | ~5 tok/s | 🎯 Cílový pro Hailo |
| **Phi-3-mini** | 3.8B | 8 GB | ~2 tok/s | ⚡ Jen s Hailo nebo swapem |

### Stažení a test modelů

```bash
# Embedding model (vždy nejdřív)
python model_tester.py --download minilm

# Lehký LLM (TinyLlama — 1.1B, ~2.2 GB)
python model_tester.py --download tinyllama

# Test všeho
python model_tester.py --test all

# Demo: RAG + LLM diagnostika
python model_tester.py --rag-demo "Modbus timeout slave 48"
```

## 🔧 Použití

### Ingest dat

```bash
# Základní použití
python ingest.py

# Přepočítat vše od začátku
python ingest.py --reindex

# Custom data
python ingest.py --data sample_data/logs.json
```

### Vyhledávání

```bash
# CLI
python query.py "CRC error stand 01"
python query.py --top-k 10 "voltage out of range"
python query.py --stand stand_03 "timeout"
python query.py --interactive          # REPL mód

# Web UI
python query.py --web
```

### Interaktivní mód

```bash
$ python query.py --interactive
💬 Interactive mode — enter queries or 'exit' to stop.

🔍 > Modbus timeout slave 48
  → Výsledky...

🔍 > :topk 10     # Změní počet výsledků
🔍 > :stand stand_03  # Filtr na stand
🔍 > :stand all   # Zruší filtr
```

## 🖥️ Nasazení na RPi 5

```bash
# Na RPi 5 s AI HAT+ 2:
git clone <repo-url> ~/tsm-rag
cd ~/tsm-rag
pip install -r requirements.txt

# Embedding na CPU (funguje hned)
python ingest.py
python query.py --web

# Až bude Hailo ready:
# 1. Nainstaluj HailoRT: sudo apt install hailo-rt
# 2. Převeď model: viz hailo_runner.py
# 3. Spusť s Hailo backendem:
python ingest.py --backend hailo
python query.py --web --backend hailo
```

## 🧪 Co dál (Phase 2)

- [ ] Reálný Log Collector (pymodbus ↔ RS485)
- [ ] Telegram notifikace z RAG výsledků
- [ ] Konverze all-MiniLM-L6-v2 na Hailo .hef
- [ ] Konverze Qwen2.5-1.5B na Hailo .hef
- [ ] Automatická diagnostika při nové chybě

## 📊 Ukázkový výstup

```
$ python query.py "Modbus timeout slave 48 register 0x10"

🔍 Hledám podobné: "Modbus timeout slave 48 register 0x10"
Výsledky (5 nejpodobnějších):
──────────────────────────────────────────
1. LOG-20260521-001 | stand_01 | 2026-05-21 08:15
   "Modbus timeout - slave 48, register 0x0010"
   Podobnost: 92.3%
   ✓ Vyřešeno: Vadný MAX485 transceiver, výměna čipu
```
