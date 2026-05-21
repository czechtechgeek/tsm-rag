"""
TSM-RAG Embedder — converts text to vector embeddings.

Supports three backends:
1. CPU: sentence-transformers/all-MiniLM-L6-v2 (default, works everywhere)
2. Hailo-10H: HailoRT with .hef model (needs AI HAT+ 2)
3. ONNX: ONNX Runtime (alternative to sentence-transformers)
"""

import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base for all embedder backends."""

    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_dim(self) -> int:
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    """CPU-based embedding using sentence-transformers (all-MiniLM-L6-v2).

    This is the default backend. Works on any machine, including RPi 5.
    Model: ~22 MB, ~800 sentences/sec on RPi 5 CPU.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DIMENSION = 384

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.MODEL_NAME
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        logger.info(f"Loading model: {self.model_name}")
        self._model = SentenceTransformer(self_model_name)
        logger.info(f"Model loaded. Dimension: {self.embed_dim()}")

    def encode(self, texts: List[str]) -> List[List[float]]:
        self._load()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_dim(self) -> int:
        return self.DIMENSION


class OnnxEmbedder(BaseEmbedder):
    """ONNX Runtime backend for sentence-transformers models.

    Alternative to the full sentence-transformers package. Useful
    when you want to avoid the HF transformers dependency or when
    you need to use a specific ONNX-optimized model.

    To use: pip install onnxruntime onnx onnxruntime-tools
    """

    DIMENSION = 384

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self._session = None
        self._tokenizer = None

    def _load(self):
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "onnxruntime + transformers required. "
                "Run: pip install onnxruntime transformers"
            )

        model_path = self.model_path or "models/onnx/all-minilm-l6-v2.onnx"
        logger.info(f"Loading ONNX model: {model_path}")
        self._session = ort.InferenceSession(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("ONNX model loaded.")

    def encode(self, texts: List[str]) -> List[List[float]]:
        self._load()
        import numpy as np

        inputs = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="np", max_length=128
        )
        outputs = self._session.run(
            None, {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])),
            }
        )
        # Mean pooling
        embeddings = self._mean_pooling(outputs[0], inputs["attention_mask"])
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.tolist()

    @staticmethod
    def _mean_pooling(token_embeddings, attention_mask):
        import numpy as np

        mask = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        return np.sum(token_embeddings * mask, axis=1) / np.sum(mask, axis=1)

    def embed_dim(self) -> int:
        return self.DIMENSION


def create_embedder(backend: str = "sentence_transformers", **kwargs) -> BaseEmbedder:
    """Factory: create the right embedder based on backend name.

    Args:
        backend: "sentence_transformers" (default), "onnx", or "hailo"
        **kwargs: passed to the embedder constructor

    Returns:
        BaseEmbedder instance
    """
    backends = {
        "sentence_transformers": SentenceTransformerEmbedder,
        "onnx": OnnxEmbedder,
    }

    if backend == "hailo":
        try:
            from rag.src.hailo_runner import HailoEmbedder
            return HailoEmbedder(**kwargs)
        except ImportError:
            logger.warning(
                "Hailo embedder not available, falling back to sentence_transformers"
            )
            return SentenceTransformerEmbedder(**kwargs)

    cls = backends.get(backend)
    if cls is None:
        logger.warning(
            f"Unknown backend '{backend}', using sentence_transformers"
        )
        cls = SentenceTransformerEmbedder

    return cls(**kwargs)
