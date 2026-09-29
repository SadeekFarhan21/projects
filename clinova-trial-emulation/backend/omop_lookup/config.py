"""
Configuration and constants for OMOP concept lookup.

Medical-specific decisions documented inline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class LookupConfig:
    """Configuration for the OMOP concept lookup tool."""

    # --- File paths ---
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    concept_file: str = "CONCEPT.csv"
    synonym_file: str = "CONCEPT_SYNONYM.csv"
    relationship_file: str = "CONCEPT_RELATIONSHIP.csv"
    db_file: str = "omop_lookup.duckdb"
    faiss_index_file: str = "concept_embeddings.faiss"
    embedding_ids_file: str = "concept_ids.npy"

    # --- Search thresholds ---
    # Exact match returns immediately with confidence 1.0
    # Fuzzy threshold: RapidFuzz score (0-100), 95+ is very high confidence
    fuzzy_high_confidence: float = 95.0
    fuzzy_min_threshold: float = 70.0
    # Semantic threshold: cosine similarity (0-1), 0.85+ is strong match
    semantic_high_confidence: float = 0.85
    semantic_min_threshold: float = 0.60

    # If best match is below this, return top-k candidates instead
    confidence_threshold: float = 0.75
    top_k_candidates: int = 5

    # --- Medical filters ---
    # OMOP standard_concept values:
    #   'S' = Standard concept (preferred for analytics)
    #   'C' = Classification concept
    #   None = Non-standard (source vocabulary)
    require_standard: bool = False  # We search broadly, then resolve

    # --- Domain-aware vocabulary preferences ---
    # When scores are tied, prefer these vocabularies by domain
    # Higher weight = more preferred
    vocabulary_preferences: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Condition": {
            "SNOMED": 1.0,
            "ICD10CM": 0.8,
            "ICD9CM": 0.6,
            "MedDRA": 0.7,
        },
        "Procedure": {
            "SNOMED": 1.0,
            "CPT4": 0.95,
            "HCPCS": 0.85,
            "ICD10PCS": 0.8,
            "ICD9Proc": 0.6,
        },
        "Drug": {
            "RxNorm": 1.0,
            "RxNorm Extension": 0.95,
            "NDC": 0.7,
            "ATC": 0.6,
        },
        "Measurement": {
            "LOINC": 1.0,
            "SNOMED": 0.8,
        },
        "Observation": {
            "SNOMED": 1.0,
            "LOINC": 0.9,
        },
        "Device": {
            "SNOMED": 1.0,
            "HCPCS": 0.8,
        },
        "Specimen": {
            "SNOMED": 1.0,
        },
        # Default for unknown domains
        "default": {
            "SNOMED": 1.0,
            "RxNorm": 0.9,
            "LOINC": 0.9,
        }
    })

    # Default vocabulary weight if not in preferences
    default_vocab_weight: float = 0.5

    # --- Embedding model ---
    # SapBERT is trained on UMLS for medical concept linking
    # Falls back to general sentence-transformers if SapBERT unavailable
    embedding_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    embedding_fallback: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 768  # SapBERT dimension

    # --- FAISS settings ---
    # Use IVF for faster approximate search on large indices
    use_ivf: bool = True
    ivf_nlist: int = 1000  # Number of clusters
    ivf_nprobe: int = 50   # Clusters to search (higher = more accurate, slower)

    # --- Performance ---
    batch_size: int = 1000  # For bulk embedding generation
    max_fuzzy_candidates: int = 1000  # Limit fuzzy comparisons

    @property
    def concept_path(self) -> Path:
        return self.data_dir / self.concept_file

    @property
    def synonym_path(self) -> Path:
        return self.data_dir / self.synonym_file

    @property
    def relationship_path(self) -> Path:
        return self.data_dir / self.relationship_file

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_file

    @property
    def faiss_path(self) -> Path:
        return self.data_dir / self.faiss_index_file

    @property
    def embedding_ids_path(self) -> Path:
        return self.data_dir / self.embedding_ids_file


# Singleton default config
DEFAULT_CONFIG = LookupConfig()
