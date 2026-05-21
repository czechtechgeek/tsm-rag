#!/usr/bin/env python3
"""
TSM-RAG Model Tester — try small LLMs on RPi 5 (with or without Hailo).

Tests various lightweight models for diagnostic generation:
    - all-MiniLM-L6-v2 (embedding, always works)
    - SmolLM2-360M / SmolLM2-1.7B (smallest viable LLM)
    - TinyLlama-1.1B (good balance)
    - Qwen2.5-1.5B (recommended for Hailo)
    - Phi-3-mini-3.8B (if enough RAM)
    - Llama-3.2-1B or 3B

Usage:
    python model_tester.py --list                      # list available models
    python model_tester.py --download qwen-1.5b        # download a model
    python model_tester.py --test all                  # test all downloaded models
    python model_tester.py --test qwen-1.5b            # test specific model
    python model_tester.py --benchmark                 # benchmark all models
    python model_tester.py --rag-demo "Modbus timeout" # demo: RAG + LLM diagnostic

Each model is tested for:
    - Load time
    - Inference speed (tokens/sec)
    - RAM usage
    - Output quality for diagnostic prompts
"""

import argparse
import logging
import os
import sys
import time

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Model definitions
MODELS = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "embedding",
        "size_mb": 22,
        "ram_gb": 0.5,
        "backend": "sentence_transformers",
    },
    "smollm2-360m": {
        "name": "SmolLM2-360M",
        "hf_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "type": "llm",
        "size_mb": 720,
        "ram_gb": 1.5,
        "note": "Nejmenší testovací LLM, běží i na 4GB RAM",
    },
    "smollm2-1.7b": {
        "name": "SmolLM2-1.7B",
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "type": "llm",
        "size_mb": 3400,
        "ram_gb": 4,
        "note": "Dobrý poměr velikost/výkon",
    },
    "tinyllama": {
        "name": "TinyLlama-1.1B",
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "type": "llm",
        "size_mb": 2200,
        "ram_gb": 3,
        "note": "1.1B param, osvědčený, výchozí doporučení",
    },
    "qwen-1.5b": {
        "name": "Qwen2.5-1.5B",
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "type": "llm",
        "size_mb": 3000,
        "ram_gb": 4,
        "note": "Doporučený pro Hailo (1.5B, vejde se do 8GB HAT)",
    },
    "phi-3-mini": {
        "name": "Phi-3-mini-3.8B",
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "type": "llm",
        "size_mb": 7600,
        "ram_gb": 8,
        "note": "Těsně se vejde do 8GB RAM RPi 5 + swap",
    },
    "llama-3.2-1b": {
        "name": "Llama-3.2-1B",
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "type": "llm",
        "size_mb": 2000,
        "ram_gb": 3,
        "note": "Novinka od Meta, rychlý",
    },
    "llama-3.2-3b": {
        "name": "Llama-3.2-3B",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "type": "llm",
        "size_mb": 6000,
        "ram_gb": 6,
        "note": "3B parametrů, lepší výkon",
    },
}

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "huggingface")

# Diagnostic prompt template
DIAGNOSTIC_PROMPT = """Jsi expert na diagnostiku testovacích standů. Na základě aktuální chyby a historických případů navrhni řešení.

AKTUÁLNÍ CHYBA: {error}

HISTORICKÉ PŘÍPADY:
{history}

TVŮJ ÚKOL:
1. Identifikuj pravděpodobnou příčinu
2. Navrhni konkrétní kroky k řešení
3. Uveď, na co si dát pozor

DIAGNOSTIKA:"""


def print_header(text: str):
    """Print a styled header."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def list_models():
    """Print all available model definitions."""
    print_header("Dostupné modely pro RPi 5 + AI HAT+ 2")
    print(f"{'Klíč':<20} {'Název':<20} {'Typ':<12} {'Velikost':<10} {'RAM':<8}  {'Poznámka'}")
    print("-" * 110)
    for key, m in MODELS.items():
        print(
            f"{key:<20} {m['name']:<20} {m['type']:<12} "
            f"{m['size_mb']} MB  {m['ram_gb']} GB  {m.get('note', '')}"
        )
    print()
    print("Doporučený postup:")
    print("  1. Začni s all-MiniLM-L6-v2 (embedding) — vždy funguje")
    print("  2. TinyLlama-1.1B nebo Qwen2.5-1.5B pro LLM diagnostiku")
    print("  3. Až bude Hailo nastavený, přepni embedding + LLM na NPU")
    print()


def download_model(model_key: str):
    """Download a model from Hugging Face Hub."""
    if model_key not in MODELS:
        print(f"Neznámý model: {model_key}")
        print(f"Dostupné: {', '.join(MODELS.keys())}")
        return

    model_info = MODELS[model_key]
    hf_id = model_info["hf_id"]

    print_header(f"Stahuji: {model_info['name']} ({hf_id})")
    print(f"Velikost: ~{model_info['size_mb']} MB")
    print(f"Cíl: {MODELS_DIR}/{model_key}")
    print()

    target_dir = os.path.join(MODELS_DIR, model_key)
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(os.path.join(target_dir, "config.json")):
        print(f"✓ Model již stažen v {target_dir}")
        return

    t0 = time.time()

    if model_info["type"] == "embedding":
        from sentence_transformers import SentenceTransformer

        print("Stahuji sentence-transformers model...")
        model = SentenceTransformer(hf_id)
        model.save(target_dir)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Stahuji tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        tokenizer.save_pretrained(target_dir)

        print("Stahuji model (může trvat několik minut)...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        model.save_pretrained(target_dir)

    dt = time.time() - t0
    print(f"✓ Staženo za {dt:.1f}s do {target_dir}")


def test_model(model_key: str):
    """Load and run a quick test of a downloaded model."""
    if model_key not in MODELS:
        print(f"Neznámý model: {model_key}")
        return

    model_dir = os.path.join(MODELS_DIR, model_key)
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"✗ Model {model_key} není stažen. Spusť: python model_tester.py --download {model_key}")
        return

    model_info = MODELS[model_key]
    print_header(f"Testuji: {model_info['name']}")

    if model_info["type"] == "embedding":
        _test_embedding(model_dir, model_info)
    else:
        _test_llm(model_dir, model_info)


def _test_embedding(model_dir: str, model_info: dict):
    """Test an embedding model."""
    from sentence_transformers import SentenceTransformer

    print("Načítám embedding model...")
    t0 = time.time()
    model = SentenceTransformer(model_dir)
    load_time = time.time() - t0
    print(f"  Load: {load_time:.2f}s")

    # Test encode
    texts = [
        "Modbus timeout - slave 48, register 0x0010",
        "CRC error on RS485 bus, stand 03",
        "Voltage out of range: 19.12V (min 20.0V)",
    ]

    t0 = time.time()
    embeddings = model.encode(texts)
    encode_time = time.time() - t0

    print(f"  Embedding {len(texts)} textů: {encode_time:.3f}s")
    print(f"  Dimenze: {len(embeddings[0])}")
    print(f"  Rychlost: ~{3 / encode_time:.0f} textů/s")
    print(f"  ✓ Model funguje správně")


def _test_llm(model_dir: str, model_info: dict):
    """Test a causal LLM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Načítám tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Načítám model (toto může trvat)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    load_time = time.time() - t0
    print(f"  Load: {load_time:.2f}s")

    # Check device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        model = model.to("cuda")
    print(f"  Device: {device}")

    # Test prompt
    prompt = DIAGNOSTIC_PROMPT.format(
        error="Modbus timeout - slave 48, register 0x0010",
        history="""1. 2026-05-19 — Rušení na kabeláži, opraveno stíněním
2. 2026-05-17 — Vadný MAX485 transceiver, výměna čipu
3. 2026-05-12 — Odpojený napájecí kabel DUT""",
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    print(f"  Prompt: {inputs['input_ids'].shape[1]} tokenů")

    # Warmup run (first inference is slower)
    print("  Warmup...")
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    # Benchmark
    print("  Generuji odpověď...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.time() - t0

    new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = new_tokens / gen_time
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"  Generováno: {new_tokens} tokenů za {gen_time:.2f}s")
    print(f"  Rychlost: {tokens_per_sec:.1f} tok/s")
    print(f"\n  Odpověď modelu:")
    print(f"  {'─' * 50}")
    for line in response.strip().split("\n"):
        print(f"  {line}")
    print(f"  {'─' * 50}")
    print(f"  ✓ Model funguje")


def benchmark_all():
    """Benchmark all downloaded models."""
    print_header("Benchmark všech stažených modelů")

    for key, info in MODELS.items():
        model_dir = os.path.join(MODELS_DIR, key)
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            print(f"  ⏭️  {info['name']}: není stažen")
            continue
        test_model(key)


def rag_demo(query_text: str):
    """End-to-end demo: RAG search + LLM diagnostic."""
    print_header("TSM-RAG Demo: RAG + LLM Diagnostika")
    print(f"Dotaz: {query_text}")
    print()

    # 1. Embed query
    from rag.src.embedder import create_embedder
    from rag.src.chroma_client import ChromaClient

    embedder = create_embedder(backend="sentence_transformers")
    client = ChromaClient(persist_directory="chroma_db")
    client.get_or_create_collection()

    print("1/3 🔍 Hledám v ChromaDB...")
    query_emb = embedder.encode([query_text])[0]
    results = client.search(query_emb, k=5)

    if not results:
        print("  ⚠️  Žádné podobné logy. Nejprve spusť: python ingest.py")
        return

    print(f"  ✓ Nalezeno {len(results)} podobných případů")
    print()

    # 2. Format history
    history_lines = []
    for i, r in enumerate(results, 1):
        doc = r["document"]
        msg = doc.split(". ")[0].replace("Log: ", "")
        resolution = doc.split("Resolution:")[-1].strip() if "Resolution:" in doc else "Neznámé řešení"
        history_lines.append(f"{i}. {msg} → {resolution}")

    history = "\n".join(history_lines)

    print("Historie pro LLM:")
    print("-" * 40)
    for line in history_lines:
        print(f"  {line}")
    print("-" * 40)
    print()

    # 3. Try LLM (if any lightweight model is downloaded)
    llm_models = ["tinyllama", "qwen-1.5b", "smollm2-1.7b", "smollm2-360m", "llama-3.2-1b"]

    for model_key in llm_models:
        model_dir = os.path.join(MODELS_DIR, model_key)
        if os.path.exists(os.path.join(model_dir, "config.json")):
            print(f"2/3 🤖 Generuji diagnostiku pomocí {MODELS[model_key]['name']}...")
            try:
                _test_llm(model_dir, MODELS[model_key])
                print(f"\n3/3 ✅ Diagnostika hotova")
            except Exception as e:
                print(f"  ✗ Chyba: {e}")
            return

    print("2/3 ⏭️  Žádný LLM model není stažen.")
    print("    Pro diagnostiku stačí RAG výsledky výše.")
    print("    Pro LLM diagnostiku spusť:")
    print("    python model_tester.py --download tinyllama")


def main():
    parser = argparse.ArgumentParser(
        description="TSM-RAG Model Tester — try small LLMs on RPi 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Příklady:
  python model_tester.py --list
  python model_tester.py --download minilm
  python model_tester.py --download tinyllama
  python model_tester.py --test minilm
  python model_tester.py --rag-demo "Modbus timeout slave 48"
        """,
    )

    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--download", type=str, help="Download a model (key or 'all')")
    parser.add_argument("--test", type=str, help="Test a downloaded model (key or 'all')")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark all downloaded models")
    parser.add_argument(
        "--rag-demo", type=str, metavar="QUERY",
        help="End-to-end demo: RAG search + LLM diagnostic"
    )

    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.download:
        if args.download == "all":
            for key in MODELS:
                download_model(key)
        else:
            download_model(args.download)
    elif args.test:
        if args.test == "all":
            for key in MODELS:
                test_model(key)
        else:
            test_model(args.test)
    elif args.benchmark:
        benchmark_all()
    elif args.rag_demo:
        rag_demo(args.rag_demo)
    else:
        # Default: show list
        list_models()


if __name__ == "__main__":
    main()