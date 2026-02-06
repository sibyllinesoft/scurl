"""Embedding generation using EmbeddingGemma-300M via ONNX."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Model configuration
MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX"
EMBEDDING_FULL_DIM = 768
EMBEDDING_DIM = 128  # Matryoshka truncation for efficiency
MAX_LENGTH = 512  # Max tokens per chunk
USE_QUANTIZED = True  # Use Q4 quantized model for ~4x speedup


class EmbeddingGemmaONNX:
    """Lightweight ONNX-based embedder using EmbeddingGemma-300M.

    Uses matryoshka representation learning to truncate embeddings to 128 dims,
    providing 6x smaller vectors with minimal accuracy loss.

    Lazy-loads model on first use to avoid startup overhead.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        """Initialize embedder.

        Args:
            model_dir: Directory to cache model files. Defaults to ~/.cache/scurl/models
            embedding_dim: Output embedding dimension (128, 256, 512, or 768).
                          Smaller dimensions are faster but slightly less accurate.
        """
        self._model_dir = model_dir or self._default_model_dir()
        self._embedding_dim = embedding_dim
        self._session = None
        self._tokenizer = None

    @staticmethod
    def _default_model_dir() -> Path:
        """Get default model cache directory."""
        # Respect XDG_CACHE_HOME if set
        cache_base = os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')
        cache = Path(cache_base) / 'scurl' / 'models'
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    def _ensure_loaded(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for embeddings. "
                "Install with: pip install onnxruntime"
            )

        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "tokenizers is required for embeddings. "
                "Install with: pip install tokenizers"
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface-hub is required for model download. "
                "Install with: pip install huggingface-hub"
            )

        # Download model files if needed
        # Use quantized model for ~4x faster inference
        model_filename = "model_q4.onnx" if USE_QUANTIZED else "model.onnx"
        data_filename = "model_q4.onnx_data" if USE_QUANTIZED else "model.onnx_data"

        model_path = hf_hub_download(
            MODEL_ID,
            subfolder="onnx",
            filename=model_filename,
            cache_dir=str(self._model_dir),
        )

        # Also download the external data file (must be in same dir as model)
        hf_hub_download(
            MODEL_ID,
            subfolder="onnx",
            filename=data_filename,
            cache_dir=str(self._model_dir),
        )

        # Load ONNX session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Use all available CPU threads
        sess_options.intra_op_num_threads = 0
        sess_options.inter_op_num_threads = 0

        self._session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider'],
        )

        # Load tokenizer
        tokenizer_path = hf_hub_download(
            MODEL_ID,
            filename="tokenizer.json",
            cache_dir=str(self._model_dir),
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_path)

        # Configure tokenizer for batched input
        self._tokenizer.enable_padding(pad_id=0, length=MAX_LENGTH)
        self._tokenizer.enable_truncation(max_length=MAX_LENGTH)

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Normalized embedding vector of shape (embedding_dim,).
        """
        self._ensure_loaded()

        # Tokenize
        encoded = self._tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        # Run inference
        outputs = self._session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )

        # Get sentence embedding (second output is pooled embedding)
        # Truncate to matryoshka dimension
        embedding = outputs[1][0, :self._embedding_dim].astype(np.float32)

        # Renormalize after truncation (important for matryoshka)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            Normalized embeddings of shape (len(texts), embedding_dim).
        """
        self._ensure_loaded()

        # Tokenize batch
        encoded_batch = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encoded_batch], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask for e in encoded_batch], dtype=np.int64
        )

        # Run inference
        outputs = self._session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )

        # Get sentence embeddings, truncate to matryoshka dimension
        embeddings = outputs[1][:, :self._embedding_dim].astype(np.float32)

        # Renormalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
        embeddings = embeddings / norms

        return embeddings

    def embed_chunks(
        self,
        text: str,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> np.ndarray:
        """Embed long text by chunking and max-pooling.

        For texts longer than the model's context window, this method:
        1. Splits text into overlapping word-based chunks
        2. Embeds each chunk
        3. Max-pools across chunks to capture strongest signals

        Args:
            text: Input text (can be arbitrarily long).
            chunk_size: Maximum words per chunk.
            overlap: Words to overlap between chunks.

        Returns:
            Single normalized embedding vector capturing the full text.
        """
        self._ensure_loaded()

        # Simple word-based chunking
        words = text.split()

        if len(words) <= chunk_size:
            return self.embed(text)

        # Create overlapping chunks
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        # Embed all chunks in batch
        embeddings = self.embed_batch(chunks)

        # Max-pool across chunks (captures strongest semantic signals)
        pooled = np.max(embeddings, axis=0)

        # Renormalize
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm

        return pooled

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._session is not None


class MockEmbedder:
    """Mock embedder for testing without model dependencies.

    Generates deterministic pseudo-random embeddings based on text hash.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self._embedding_dim = embedding_dim

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding."""
        # Use text hash as seed for reproducibility
        seed = hash(text) % (2**32)
        rng = np.random.default_rng(seed)
        embedding = rng.standard_normal(self._embedding_dim).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings for batch."""
        return np.array([self.embed(t) for t in texts])

    def embed_chunks(
        self,
        text: str,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> np.ndarray:
        """Generate mock embedding for long text."""
        return self.embed(text)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def is_loaded(self) -> bool:
        return True
