"""
OMOP Concept Lookup Tool

A high-accuracy, low-latency tool for mapping free-text medical terms
to OMOP/Athena concept IDs.

Agent-Friendly Interface:
    >>> from omop_lookup import lookup, batch_lookup, initialize
    >>>
    >>> # Initialize database (required once)
    >>> initialize()
    >>>
    >>> # Single lookup
    >>> result = lookup("heart attack")
    >>> if result.success:
    >>>     print(f"Matched: {result.best_match.concept_name}")
    >>>     print(f"Concept ID: {result.best_match.get_analytics_concept_id()}")
    >>>
    >>> # Batch lookup from file
    >>> results = batch_lookup("queries.txt")

Features:
    - Exact match on concept names and synonyms
    - Fuzzy matching (RapidFuzz)
    - Semantic search (SapBERT + FAISS)
    - Automatic resolution to standard concepts via "Maps To"
    - Domain-aware vocabulary preferences
    - Confidence scoring with top-k fallback
"""

import logging
import time
import csv
from pathlib import Path
from typing import Optional, List, Union

from .config import LookupConfig, DEFAULT_CONFIG
from .models import (
    ConceptMatch,
    LookupResult,
    BatchLookupResult,
    MatchType,
    ResolutionStatus
)
from .database import OMOPDatabase, get_database
from .search import ConceptSearcher, get_searcher
from .normalize import normalize_text

__version__ = "1.0.0"
__all__ = [
    # Main functions
    "initialize",
    "lookup",
    "batch_lookup",
    "build_semantic_index",
    # Classes
    "LookupConfig",
    "ConceptMatch",
    "LookupResult",
    "BatchLookupResult",
    "MatchType",
    "ResolutionStatus",
    # Low-level access
    "get_database",
    "get_searcher",
]

logger = logging.getLogger(__name__)


def initialize(
    config: Optional[LookupConfig] = None,
    force_rebuild: bool = False,
    update_normalized: bool = True
) -> None:
    """
    Initialize the OMOP lookup database.

    This must be called once before using lookup functions.
    Loads OMOP vocabulary CSV files into DuckDB and creates
    optimized lookup tables.

    Args:
        config: Optional custom configuration
        force_rebuild: If True, recreate database from CSVs even if exists
        update_normalized: If True, update normalized text columns (slow but required for exact match)

    Example:
        >>> from omop_lookup import initialize
        >>> initialize()  # First-time setup
        >>> initialize(force_rebuild=True)  # Rebuild from scratch
    """
    cfg = config or DEFAULT_CONFIG

    logger.info("Initializing OMOP lookup database...")

    db = get_database(cfg)
    db.initialize(force_rebuild=force_rebuild)

    if update_normalized or force_rebuild:
        db.update_normalized_text()

    stats = db.get_stats()
    logger.info(f"Database initialized: {stats}")

    print(f"OMOP Lookup initialized successfully!")
    print(f"  Concepts: {stats.get('concept', 0):,}")
    print(f"  Synonyms: {stats.get('concept_synonym', 0):,}")
    print(f"  Mappings: {stats.get('maps_to_lookup', 0):,}")
    print(f"  Search entries: {stats.get('concept_search', 0):,}")


def lookup(
    query: str,
    domain: Optional[str] = None,
    top_k: int = 10,
    enable_semantic: bool = False,  # Disabled by default for speed
    config: Optional[LookupConfig] = None
) -> LookupResult:
    """
    Look up a medical term and return matching OMOP concepts.

    This is the main entry point for single-query lookups.

    Args:
        query: Free-text medical term (e.g., "heart attack", "knee surgery")
        domain: Optional domain filter ("Condition", "Procedure", "Drug", etc.)
        top_k: Number of candidates to return if confidence is low
        enable_semantic: Whether to use semantic search (requires built index)
        config: Optional custom configuration

    Returns:
        LookupResult with best_match and/or candidates

    Example:
        >>> result = lookup("myocardial infarction", domain="Condition")
        >>> if result.success:
        >>>     match = result.best_match
        >>>     print(f"Concept ID: {match.concept_id}")
        >>>     print(f"Name: {match.concept_name}")
        >>>     print(f"Confidence: {match.confidence:.2f}")
        >>>     print(f"Analytics ID: {match.get_analytics_concept_id()}")

    Agent Integration:
        The returned LookupResult can be serialized to JSON:
        >>> import json
        >>> json.dumps(result.to_dict())
    """
    cfg = config or DEFAULT_CONFIG
    searcher = get_searcher(cfg, enable_semantic=enable_semantic)

    return searcher.search(query, domain_filter=domain, top_k=top_k)


def batch_lookup(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    domain: Optional[str] = None,
    delimiter: str = ",",
    query_column: int = 0,
    has_header: bool = True,
    enable_semantic: bool = True,
    config: Optional[LookupConfig] = None
) -> BatchLookupResult:
    """
    Process multiple queries from a file.

    Reads queries from a .txt or .csv file and returns results for all.

    Args:
        input_path: Path to input file (.txt with one query per line, or .csv)
        output_path: Optional path to write results CSV
        domain: Optional domain filter for all queries
        delimiter: CSV delimiter (default comma)
        query_column: Column index containing queries (0-based, for CSV)
        has_header: Whether CSV has a header row
        enable_semantic: Whether to use semantic search
        config: Optional custom configuration

    Returns:
        BatchLookupResult with all results and summary statistics

    Example:
        >>> # From text file (one query per line)
        >>> results = batch_lookup("queries.txt", "results.csv")
        >>> print(f"Success rate: {results.successful_queries}/{results.total_queries}")
        >>>
        >>> # From CSV file
        >>> results = batch_lookup("data.csv", query_column=2)
    """
    cfg = config or DEFAULT_CONFIG
    searcher = get_searcher(cfg, enable_semantic=enable_semantic)

    input_path = Path(input_path)
    queries = []

    # Read queries from file
    if input_path.suffix.lower() == ".txt":
        with open(input_path, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        with open(input_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            if has_header:
                next(reader, None)  # Skip header
            for row in reader:
                if len(row) > query_column:
                    query = row[query_column].strip()
                    if query:
                        queries.append(query)

    # Process queries
    start_time = time.time()
    results: List[LookupResult] = []
    successful = 0

    for i, query in enumerate(queries):
        result = searcher.search(query, domain_filter=domain)
        results.append(result)
        if result.success:
            successful += 1

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(queries)} queries...")

    total_time = (time.time() - start_time) * 1000

    batch_result = BatchLookupResult(
        results=results,
        total_queries=len(queries),
        successful_queries=successful,
        total_time_ms=total_time
    )

    # Write output if requested
    if output_path:
        _write_batch_results(batch_result, output_path, queries)
        logger.info(f"Results written to {output_path}")

    return batch_result


def _write_batch_results(
    batch_result: BatchLookupResult,
    output_path: Union[str, Path],
    queries: List[str]
) -> None:
    """Write batch results to CSV file."""
    output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "query",
            "success",
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "confidence",
            "match_type",
            "resolution_status",
            "analytics_concept_id",
            "standard_concept_name",
            "warnings"
        ])

        # Data rows
        for query, result in zip(queries, batch_result.results):
            match = result.best_match
            if match:
                writer.writerow([
                    query,
                    result.success,
                    match.concept_id,
                    match.concept_name,
                    match.domain_id,
                    match.vocabulary_id,
                    f"{match.confidence:.4f}",
                    match.match_type.value,
                    match.resolution_status.value,
                    match.get_analytics_concept_id(),
                    match.standard_concept_name or "",
                    "; ".join(result.warnings)
                ])
            else:
                writer.writerow([
                    query,
                    False,
                    "", "", "", "", "", "", "", "", "",
                    "; ".join(result.warnings)
                ])


def build_semantic_index(
    domain_filter: Optional[str] = None,
    config: Optional[LookupConfig] = None
) -> None:
    """
    Build the FAISS index for semantic search.

    This is a potentially long-running operation (hours for full vocabulary).
    Only needs to be run once unless vocabulary is updated.

    Args:
        domain_filter: Optional domain to limit index (reduces size/time)
        config: Optional custom configuration

    Example:
        >>> # Build full index (slow but comprehensive)
        >>> build_semantic_index()
        >>>
        >>> # Build for specific domain (faster)
        >>> build_semantic_index(domain_filter="Condition")
    """
    cfg = config or DEFAULT_CONFIG

    from .embeddings import get_semantic_search

    print("Building semantic search index...")
    print("This may take several hours for the full vocabulary.")
    print("Consider using domain_filter to limit scope.")

    semantic = get_semantic_search(cfg)
    db = get_database(cfg)

    semantic.build_index(db, domain_filter=domain_filter)

    print("Semantic search index built successfully!")


def get_stats(config: Optional[LookupConfig] = None) -> dict:
    """
    Get database statistics.

    Returns:
        Dictionary with table counts
    """
    cfg = config or DEFAULT_CONFIG
    db = get_database(cfg)
    return db.get_stats()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
