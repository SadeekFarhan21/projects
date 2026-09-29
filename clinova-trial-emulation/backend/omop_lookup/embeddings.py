"""
Embedding generation and FAISS index management for semantic search.

Uses SapBERT (Self-Alignment Pre-training for BERT) which is specifically
trained on UMLS medical concepts for concept linking tasks.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import warnings

from .config import LookupConfig, DEFAULT_CONFIG
from .database import OMOPDatabase, get_database

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_torch = None
_transformers = None
_faiss = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _import_transformers():
    global _transformers
    if _transformers is None:
        from transformers import AutoTokenizer, AutoModel
        _transformers = {"AutoTokenizer": AutoTokenizer, "AutoModel": AutoModel}
    return _transformers


def _import_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


class EmbeddingModel:
    """
    Wrapper for SapBERT or fallback embedding model.

    SapBERT is preferred for medical concept linking as it was trained
    specifically on UMLS concept pairs, learning semantic similarity
    between medical terms.
    """

    def __init__(self, config: LookupConfig = DEFAULT_CONFIG):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize the embedding model.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True

        torch = _import_torch()
        transformers = _import_transformers()

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Try to load SapBERT first
        model_name = self.config.embedding_model
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.tokenizer = transformers["AutoTokenizer"].from_pretrained(model_name)
            self.model = transformers["AutoModel"].from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            logger.info(f"Successfully loaded {model_name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            logger.info(f"Falling back to {self.config.embedding_fallback}")

            try:
                model_name = self.config.embedding_fallback
                self.tokenizer = transformers["AutoTokenizer"].from_pretrained(model_name)
                self.model = transformers["AutoModel"].from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                self._initialized = True
                logger.info(f"Successfully loaded fallback model {model_name}")
                return True

            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                return False

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts into embeddings.

        Uses mean pooling over token embeddings (standard for SapBERT).

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize embedding model")

        torch = _import_torch()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,  # Medical terms are usually short
                    return_tensors="pt"
                ).to(self.device)

                # Get embeddings
                outputs = self.model(**inputs)

                # Mean pooling over tokens (excluding padding)
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state

                # Expand attention mask for broadcasting
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

                # Sum embeddings and divide by number of non-padding tokens
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask

                # Normalize embeddings (important for cosine similarity)
                mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

                all_embeddings.append(mean_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding.

        Args:
            text: Text to encode

        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.encode([text])[0]


class FAISSIndex:
    """
    FAISS index for fast approximate nearest neighbor search.

    Uses IVF (Inverted File Index) for scalability with millions of concepts.
    """

    def __init__(self, config: LookupConfig = DEFAULT_CONFIG):
        self.config = config
        self.index = None
        self.concept_ids: Optional[np.ndarray] = None
        self._initialized = False

    def is_built(self) -> bool:
        """Check if the FAISS index exists on disk."""
        return (
            self.config.faiss_path.exists() and
            self.config.embedding_ids_path.exists()
        )

    def load(self) -> bool:
        """
        Load existing FAISS index from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.is_built():
            return False

        faiss = _import_faiss()

        try:
            logger.info(f"Loading FAISS index from {self.config.faiss_path}")
            self.index = faiss.read_index(str(self.config.faiss_path))
            self.concept_ids = np.load(str(self.config.embedding_ids_path))
            self._initialized = True
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def build(
        self,
        db: OMOPDatabase,
        embedding_model: EmbeddingModel,
        domain_filter: Optional[str] = None
    ) -> None:
        """
        Build FAISS index from database concepts.

        This is a potentially long-running operation (hours for millions of concepts).

        Args:
            db: Database instance
            embedding_model: Initialized embedding model
            domain_filter: Optional domain to filter concepts
        """
        faiss = _import_faiss()

        logger.info("Building FAISS index (this may take a while)...")

        all_concept_ids = []
        all_embeddings = []

        batch_num = 0
        for batch in db.get_all_concepts_for_embedding(
            batch_size=self.config.batch_size,
            domain_filter=domain_filter
        ):
            batch_num += 1

            # Extract texts for embedding
            concept_ids = [row[0] for row in batch]
            texts = [row[2] if row[2] else row[1] for row in batch]  # Prefer normalized

            # Generate embeddings
            embeddings = embedding_model.encode(texts, batch_size=32)

            all_concept_ids.extend(concept_ids)
            all_embeddings.append(embeddings)

            if batch_num % 10 == 0:
                total_embedded = len(all_concept_ids)
                logger.info(f"Embedded {total_embedded:,} concepts...")

        # Stack all embeddings
        all_embeddings = np.vstack(all_embeddings).astype(np.float32)
        self.concept_ids = np.array(all_concept_ids, dtype=np.int64)

        logger.info(f"Total embeddings: {len(self.concept_ids):,}")

        # Build FAISS index
        embedding_dim = all_embeddings.shape[1]

        if self.config.use_ivf and len(self.concept_ids) > 10000:
            # Use IVF for large indices
            logger.info(f"Building IVF index with {self.config.ivf_nlist} clusters...")

            quantizer = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine for normalized)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                embedding_dim,
                self.config.ivf_nlist,
                faiss.METRIC_INNER_PRODUCT
            )

            # Train the index
            logger.info("Training IVF index...")
            self.index.train(all_embeddings)

            # Add vectors
            logger.info("Adding vectors to index...")
            self.index.add(all_embeddings)

            # Set search parameters
            self.index.nprobe = self.config.ivf_nprobe

        else:
            # Use flat index for smaller datasets
            logger.info("Building flat index...")
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.index.add(all_embeddings)

        self._initialized = True
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")

        # Save to disk
        self.save()

    def save(self) -> None:
        """Save FAISS index to disk."""
        if not self._initialized:
            raise RuntimeError("Index not initialized")

        faiss = _import_faiss()

        logger.info(f"Saving FAISS index to {self.config.faiss_path}")
        faiss.write_index(self.index, str(self.config.faiss_path))
        np.save(str(self.config.embedding_ids_path), self.concept_ids)
        logger.info("FAISS index saved")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors.

        Args:
            query_embedding: Query embedding vector
            k: Number of neighbors to return

        Returns:
            List of (concept_id, similarity_score) tuples
        """
        if not self._initialized:
            if not self.load():
                raise RuntimeError("FAISS index not available. Run build_index first.")

        # Ensure query is 2D and float32
        query = query_embedding.astype(np.float32).reshape(1, -1)

        # Search
        scores, indices = self.index.search(query, k)

        # Map indices to concept IDs
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # -1 indicates no result
                concept_id = int(self.concept_ids[idx])
                results.append((concept_id, float(score)))

        return results


class SemanticSearch:
    """
    High-level semantic search interface combining embedding model and FAISS.
    """

    def __init__(self, config: LookupConfig = DEFAULT_CONFIG):
        self.config = config
        self.embedding_model = EmbeddingModel(config)
        self.faiss_index = FAISSIndex(config)
        self._available = None

    def is_available(self) -> bool:
        """Check if semantic search is available (model loaded, index built)."""
        if self._available is not None:
            return self._available

        try:
            # Check if we can load the model
            if not self.embedding_model.initialize():
                self._available = False
                return False

            # Check if index exists
            if not self.faiss_index.is_built():
                logger.warning("FAISS index not built. Run build_index() first.")
                self._available = False
                return False

            # Try to load index
            if not self.faiss_index.load():
                self._available = False
                return False

            self._available = True
            return True

        except Exception as e:
            logger.warning(f"Semantic search not available: {e}")
            self._available = False
            return False

    def build_index(
        self,
        db: Optional[OMOPDatabase] = None,
        domain_filter: Optional[str] = None
    ) -> None:
        """
        Build the FAISS index for semantic search.

        Args:
            db: Database instance (uses default if not provided)
            domain_filter: Optional domain filter
        """
        if db is None:
            db = get_database(self.config)

        if not self.embedding_model.initialize():
            raise RuntimeError("Failed to initialize embedding model")

        self.faiss_index.build(db, self.embedding_model, domain_filter)
        self._available = True

    def search(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Perform semantic search for a query.

        Args:
            query: Search query (will be embedded)
            k: Number of results to return

        Returns:
            List of (concept_id, similarity_score) tuples
        """
        if not self.is_available():
            raise RuntimeError("Semantic search not available")

        # Embed query
        query_embedding = self.embedding_model.encode_single(query)

        # Search FAISS
        return self.faiss_index.search(query_embedding, k)


# Module-level singleton
_semantic_search: Optional[SemanticSearch] = None


def get_semantic_search(config: LookupConfig = DEFAULT_CONFIG) -> SemanticSearch:
    """Get or create the singleton semantic search instance."""
    global _semantic_search
    if _semantic_search is None:
        _semantic_search = SemanticSearch(config)
    return _semantic_search
