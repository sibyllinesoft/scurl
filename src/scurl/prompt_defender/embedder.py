"""Embedding generation using MiniLM-L6-v2 via ONNX."""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

# Model configuration - MiniLM-L6-v2 (22M params, 86MB, ~15ms inference)
MODEL_ID = "Qdrant/all-MiniLM-L6-v2-onnx"
EMBEDDING_DIM = 384  # MiniLM output dimension
MAX_LENGTH = 256  # Max tokens per chunk


class MiniLMEmbedder:
    """Lightweight ONNX-based embedder using all-MiniLM-L6-v2.

    Fast (~15ms) sentence embeddings with 384 dimensions.
    Lazy-loads model on first use to avoid startup overhead.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
    ):
        """Initialize embedder.

        Args:
            model_dir: Directory to cache model files. Defaults to ~/.cache/scurl/models
        """
        self._model_dir = model_dir or self._default_model_dir()
        self._embedding_dim = EMBEDDING_DIM
        self._session = None
        self._tokenizer = None

    @staticmethod
    def _default_model_dir() -> Path:
        """Get default model cache directory."""
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

        # Download model file (single file, no external data)
        model_path = hf_hub_download(
            MODEL_ID,
            filename="model.onnx",
            cache_dir=str(self._model_dir),
        )

        # Load ONNX session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
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

        # Configure tokenizer
        self._tokenizer.enable_padding(pad_id=0, length=MAX_LENGTH)
        self._tokenizer.enable_truncation(max_length=MAX_LENGTH)

    def _mean_pooling(
        self, last_hidden_state: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """Apply mean pooling to get sentence embeddings.

        Args:
            last_hidden_state: Shape (batch, seq_len, hidden_dim)
            attention_mask: Shape (batch, seq_len)

        Returns:
            Pooled embeddings of shape (batch, hidden_dim)
        """
        # Expand attention mask for broadcasting
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)

        # Sum embeddings weighted by attention mask
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)

        # Divide by number of non-padding tokens
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        return sum_embeddings / sum_mask

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Normalized embedding vector of shape (384,).
        """
        self._ensure_loaded()

        # Tokenize
        encoded = self._tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        # Run inference
        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        # Mean pooling over sequence
        embedding = self._mean_pooling(outputs[0], attention_mask)[0]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            Normalized embeddings of shape (len(texts), 384).
        """
        self._ensure_loaded()

        # Tokenize batch
        encoded_batch = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encoded_batch], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask for e in encoded_batch], dtype=np.int64
        )
        token_type_ids = np.zeros_like(input_ids)

        # Run inference
        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        # Mean pooling
        embeddings = self._mean_pooling(outputs[0], attention_mask)

        # Normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def embed_chunks(
        self,
        text: str,
        chunk_size: int = 200,
        overlap: int = 30,
    ) -> np.ndarray:
        """Embed long text by chunking and max-pooling.

        Args:
            text: Input text (can be arbitrarily long).
            chunk_size: Maximum words per chunk.
            overlap: Words to overlap between chunks.

        Returns:
            Single normalized embedding vector.
        """
        self._ensure_loaded()

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

        # Max-pool across chunks
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


# Alias for backward compatibility
EmbeddingGemmaONNX = MiniLMEmbedder


class MockEmbedder:
    """Mock embedder for testing without model dependencies."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self._embedding_dim = embedding_dim

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding."""
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
        chunk_size: int = 200,
        overlap: int = 30,
    ) -> np.ndarray:
        """Generate mock embedding for long text."""
        return self.embed(text)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def is_loaded(self) -> bool:
        return True
