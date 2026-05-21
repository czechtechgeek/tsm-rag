"""
TSM-RAG Hailo-10H Runner — inference on AI HAT+ 2.

This module provides a Hailo-compatible embedder that runs
all-MiniLM-L6-v2 (converted to .hef format) on the Hailo-10H NPU.

Prerequisites (run on RPi 5 with AI HAT+ 2):
    sudo apt install hailo-rt
    pip install hailort

Model conversion (on PC with Hailo Dataflow Compiler):
    hailo dataflow compile --model all-minilm-l6-v2.onnx --output models/hailo/minilm-l6.hef

For now, this module acts as a stub/prototype. The actual Hailo
driver and runtime will be set up once the AI HAT+ 2 is configured
on the RPi 5.
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

HAILO_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "hailo")
DEFAULT_HEF = os.path.join(HAILO_MODELS_DIR, "minilm-l6.hef")

# Embedding dimension of all-MiniLM-L6-v2
EMBED_DIM = 384


def is_hailo_available() -> bool:
    """Check if Hailo runtime and hardware are available.

    Returns:
        True if a Hailo device is detected
    """
    try:
        import hailort
        devices = hailort.Device.scan()
        return len(devices) > 0
    except (ImportError, Exception):
        return False


class HailoEmbedder:
    """Hailo-10H embedder using pre-compiled .hef model.

    Usage:
        embedder = HailoEmbedder()
        vectors = embedder.encode(["text1", "text2"])
    """

    def __init__(self, hef_path: str = None):
        self.hef_path = hef_path or DEFAULT_HEF
        self._model = None
        self._configured_model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import hailort
        except ImportError:
            raise ImportError(
                "hailort not installed. Install HailoRT on RPi 5:\n"
                "  sudo apt install hailo-rt\n"
                "  pip install hailort"
            )

        if not os.path.exists(self.hef_path):
            logger.warning(
                f"Hailo model not found at {self.hef_path}. "
                "Falling back to CPU would require setting up model conversion first."
            )
            raise FileNotFoundError(
                f"Hailo .hef model not found: {self.hef_path}\n"
                "Convert all-MiniLM-L6-v2 to Hailo format:\n"
                "  1. Export to ONNX: transformers.onnx → model.onnx\n"
                "  2. Compile with Hailo Dataflow Compiler:\n"
                "     hailo dataflow compile --model model.onnx --output models/hailo/minilm-l6.hef"
            )

        logger.info(f"Loading Hailo model: {self.hef_path}")
        self._model = hailort.HefFile(self.hef_path)
        self._configured_model = self._model.configure()
        logger.info("Hailo model loaded and configured")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using Hailo NPU.

        NOTE: This is a prototype. Sentence-transformers / ONNX
        preprocessing (tokenization) still happens on CPU. Only the
        neural network forward pass runs on Hailo.
        """
        self._load()
        # TODO: Implement tokenization + Hailo inference pipeline
        # For now, log the intent
        logger.info(
            f"Hailo inference requested for {len(texts)} texts. "
            "Full implementation pending Hailo-10H setup on RPi 5."
        )
        raise NotImplementedError(
            "Hailo inference is not yet implemented. "
            "Use SentenceTransformerEmbedder for CPU-based testing."
        )

    def embed_dim(self) -> int:
        return EMBED_DIM


class HailoLLMRunner:
    """Run lightweight LLM (Qwen 2.5 1.5B, Phi-3-mini) on Hailo-10H.

    The Hailo-10H has 8 GB of dedicated RAM, enough for 1.5B-3B
    parameter models when compiled to Hailo format.

    NOTE: This is a prototype. LLM inference on Hailo requires
    the Hailo Dataflow Compiler and specific model conversion steps.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(HAILO_MODELS_DIR, "")
        self._model = None

    def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.1
    ) -> str:
        """Generate text using Hailo-10H NPU.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)

        Returns:
            Generated text
        """
        raise NotImplementedError(
            "Hailo LLM inference is not yet implemented. "
            "Use model_tester.py for CPU-based testing of small LLMs."
        )


def get_hailo_info() -> dict:
    """Return info about detected Hailo device.

    Returns:
        Dict with device info or error message
    """
    try:
        import hailort

        devices = hailort.Device.scan()
        if devices:
            info = []
            for dev in devices:
                info.append({
                    "id": dev.device_id,
                    "type": dev.device_type,
                })
            return {"devices": info, "available": True}
        return {"devices": [], "available": False, "message": "No Hailo device found"}
    except ImportError:
        return {"devices": [], "available": False, "message": "hailort not installed"}
    except Exception as e:
        return {"devices": [], "available": False, "message": str(e)}
