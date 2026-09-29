"""
Data models for OMOP concept lookup results.

These models are designed to be agent-friendly with clear, typed fields.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class MatchType(Enum):
    """How the concept was matched."""
    EXACT_NAME = "exact_name"
    EXACT_SYNONYM = "exact_synonym"
    FUZZY_NAME = "fuzzy_name"
    FUZZY_SYNONYM = "fuzzy_synonym"
    SEMANTIC = "semantic"


class ResolutionStatus(Enum):
    """Status of standard concept resolution."""
    ALREADY_STANDARD = "already_standard"
    MAPPED_TO_STANDARD = "mapped_to_standard"
    NO_MAPPING_AVAILABLE = "no_mapping_available"


@dataclass
class ConceptMatch:
    """
    A single matched concept with full metadata.

    For agent consumption: check `confidence` and `resolution_status` to determine
    whether to trust this match or request human review.
    """
    # Core identifiers
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: Optional[str]  # 'S', 'C', or None

    # Match quality
    confidence: float  # 0.0 to 1.0
    match_type: MatchType
    matched_text: str  # The text that was actually matched (name or synonym)

    # Standard resolution (critical for OMOP analytics)
    resolution_status: ResolutionStatus
    standard_concept_id: Optional[int] = None  # If resolved via Maps To
    standard_concept_name: Optional[str] = None

    # Additional context
    concept_code: Optional[str] = None  # Source vocabulary code (e.g., SNOMED code)

    def is_high_confidence(self, threshold: float = 0.75) -> bool:
        """Check if this match meets confidence threshold."""
        return self.confidence >= threshold

    def is_usable_for_analytics(self) -> bool:
        """
        Check if this concept can be used for OMOP analytics.

        Returns True if:
        - Concept is already standard, OR
        - Concept was successfully mapped to a standard concept
        """
        return self.resolution_status in (
            ResolutionStatus.ALREADY_STANDARD,
            ResolutionStatus.MAPPED_TO_STANDARD
        )

    def get_analytics_concept_id(self) -> Optional[int]:
        """
        Get the concept_id to use for OMOP analytics.

        Returns the standard concept ID (either self or mapped target).
        Returns None if no standard mapping is available.
        """
        if self.resolution_status == ResolutionStatus.ALREADY_STANDARD:
            return self.concept_id
        elif self.resolution_status == ResolutionStatus.MAPPED_TO_STANDARD:
            return self.standard_concept_id
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "domain_id": self.domain_id,
            "vocabulary_id": self.vocabulary_id,
            "concept_class_id": self.concept_class_id,
            "standard_concept": self.standard_concept,
            "confidence": round(self.confidence, 4),
            "match_type": self.match_type.value,
            "matched_text": self.matched_text,
            "resolution_status": self.resolution_status.value,
            "standard_concept_id": self.standard_concept_id,
            "standard_concept_name": self.standard_concept_name,
            "concept_code": self.concept_code,
            "analytics_concept_id": self.get_analytics_concept_id(),
            "usable_for_analytics": self.is_usable_for_analytics(),
        }


@dataclass
class LookupResult:
    """
    Complete result of a concept lookup query.

    Agent-friendly interface:
    - If `success` is True and `best_match` exists with high confidence, use it
    - If confidence is low, review `candidates` for alternatives
    - Check `warnings` for any data quality concerns
    """
    # Query info
    query: str
    normalized_query: str
    domain_filter: Optional[str]

    # Results
    success: bool
    best_match: Optional[ConceptMatch] = None
    candidates: List[ConceptMatch] = field(default_factory=list)

    # Metadata
    search_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "domain_filter": self.domain_filter,
            "success": self.success,
            "best_match": self.best_match.to_dict() if self.best_match else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "search_time_ms": round(self.search_time_ms, 2),
            "warnings": self.warnings,
        }


@dataclass
class BatchLookupResult:
    """Result of batch lookup for multiple queries."""
    results: List[LookupResult]
    total_queries: int
    successful_queries: int
    total_time_ms: float

    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "total_time_ms": round(self.total_time_ms, 2),
            "success_rate": round(self.successful_queries / max(self.total_queries, 1), 4),
        }
