"""
Hybrid search implementation with tiered approach.

Search tiers (with early exit):
1. Exact match on normalized text -> confidence 1.0, return immediately
2. Fuzzy match on token-filtered candidates -> if score > 0.95, return
3. Semantic search via FAISS -> combine with fuzzy scores

After matching, resolve non-standard concepts to standard via "Maps To".
"""

import logging
import time
from typing import Optional, List, Dict, Any, Tuple

from .config import LookupConfig, DEFAULT_CONFIG
from .models import ConceptMatch, LookupResult, MatchType, ResolutionStatus
from .normalize import normalize_text, tokenize, calculate_token_overlap
from .database import OMOPDatabase, get_database

logger = logging.getLogger(__name__)

# Lazy import for RapidFuzz
_rapidfuzz = None


def _import_rapidfuzz():
    global _rapidfuzz
    if _rapidfuzz is None:
        from rapidfuzz import fuzz, process
        _rapidfuzz = {"fuzz": fuzz, "process": process}
    return _rapidfuzz


class ConceptSearcher:
    """
    Main search engine implementing tiered hybrid search.

    Search strategy:
    1. Exact match (O(1) database lookup)
    2. Token-overlap filter + fuzzy match (reduce candidates, then RapidFuzz)
    3. Semantic search (FAISS nearest neighbors)

    Results are resolved to standard concepts and ranked by confidence.
    """

    def __init__(
        self,
        config: LookupConfig = DEFAULT_CONFIG,
        db: Optional[OMOPDatabase] = None,
        enable_semantic: bool = True
    ):
        self.config = config
        self.db = db or get_database(config)
        self.enable_semantic = enable_semantic
        self._semantic_search = None

    def _get_semantic_search(self):
        """Lazy-load semantic search to avoid import overhead."""
        if self._semantic_search is None and self.enable_semantic:
            try:
                from .embeddings import get_semantic_search
                self._semantic_search = get_semantic_search(self.config)
            except ImportError as e:
                logger.warning(f"Semantic search not available: {e}")
                self.enable_semantic = False
        return self._semantic_search

    def search(
        self,
        query: str,
        domain_filter: Optional[str] = None,
        top_k: int = 5
    ) -> LookupResult:
        """
        Search for concepts matching the query.

        Implements tiered search with early exit for efficiency.

        Args:
            query: Free-text search query (e.g., "heart attack", "knee surgery")
            domain_filter: Optional domain filter (e.g., "Condition", "Procedure")
            top_k: Number of candidates to return if confidence is low

        Returns:
            LookupResult with best match and/or candidates
        """
        start_time = time.time()

        # Normalize query
        normalized_query = normalize_text(query)
        tokens = tokenize(normalized_query)

        warnings = []

        # --- Tier 1: Exact Match ---
        exact_results = self._exact_match(normalized_query, domain_filter)

        if exact_results:
            # Found exact match - resolve and return
            best = self._select_best_match(exact_results, domain_filter)
            match = self._create_match(
                best,
                confidence=1.0,
                match_type=MatchType.EXACT_NAME if best["source_type"] == "name" else MatchType.EXACT_SYNONYM,
                matched_text=best["search_text"]
            )

            return LookupResult(
                query=query,
                normalized_query=normalized_query,
                domain_filter=domain_filter,
                success=True,
                best_match=match,
                candidates=[],
                search_time_ms=(time.time() - start_time) * 1000,
                warnings=warnings
            )

        # --- Tier 2: Fuzzy Match ---
        fuzzy_results = self._fuzzy_match(normalized_query, tokens, domain_filter)

        if fuzzy_results:
            best_fuzzy = fuzzy_results[0]

            if best_fuzzy["score"] >= self.config.fuzzy_high_confidence:
                # High confidence fuzzy match - return immediately
                match = self._create_match(
                    best_fuzzy["concept"],
                    confidence=best_fuzzy["score"] / 100.0,
                    match_type=MatchType.FUZZY_NAME if best_fuzzy["concept"]["source_type"] == "name" else MatchType.FUZZY_SYNONYM,
                    matched_text=best_fuzzy["concept"]["search_text"]
                )

                return LookupResult(
                    query=query,
                    normalized_query=normalized_query,
                    domain_filter=domain_filter,
                    success=True,
                    best_match=match,
                    candidates=[],
                    search_time_ms=(time.time() - start_time) * 1000,
                    warnings=warnings
                )

        # --- Tier 3: Semantic Search ---
        semantic_results = []
        if self.enable_semantic:
            semantic_search = self._get_semantic_search()
            if semantic_search and semantic_search.is_available():
                try:
                    semantic_results = self._semantic_match(
                        normalized_query,
                        domain_filter,
                        top_k=top_k * 2  # Get more for merging
                    )
                except Exception as e:
                    logger.warning(f"Semantic search failed: {e}")
                    warnings.append(f"Semantic search unavailable: {str(e)}")
            else:
                warnings.append("Semantic search index not built")

        # --- Combine and Rank Results ---
        all_candidates = self._merge_and_rank(
            fuzzy_results,
            semantic_results,
            domain_filter,
            top_k
        )

        if not all_candidates:
            return LookupResult(
                query=query,
                normalized_query=normalized_query,
                domain_filter=domain_filter,
                success=False,
                best_match=None,
                candidates=[],
                search_time_ms=(time.time() - start_time) * 1000,
                warnings=warnings + ["No matches found"]
            )

        # Create ConceptMatch objects
        matches = [
            self._create_match(
                c["concept"],
                confidence=c["combined_score"],
                match_type=c["match_type"],
                matched_text=c["concept"].get("search_text", c["concept"]["concept_name"])
            )
            for c in all_candidates
        ]

        best_match = matches[0]
        candidates = matches[1:top_k] if len(matches) > 1 else []

        # Determine success based on confidence threshold
        success = best_match.confidence >= self.config.confidence_threshold

        if not success:
            warnings.append(
                f"Best match confidence ({best_match.confidence:.2f}) below threshold "
                f"({self.config.confidence_threshold}). Review candidates."
            )

        return LookupResult(
            query=query,
            normalized_query=normalized_query,
            domain_filter=domain_filter,
            success=success,
            best_match=best_match,
            candidates=candidates,
            search_time_ms=(time.time() - start_time) * 1000,
            warnings=warnings
        )

    def _exact_match(
        self,
        normalized_query: str,
        domain_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Find exact matches in database."""
        return self.db.exact_match(normalized_query, domain_filter, limit=10)

    def _fuzzy_match(
        self,
        normalized_query: str,
        tokens: List[str],
        domain_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Find fuzzy matches using RapidFuzz.

        First filters candidates by token overlap to reduce search space,
        then applies fuzzy matching on the filtered set.
        """
        rapidfuzz = _import_rapidfuzz()

        # Get candidates with token overlap
        candidates = self.db.get_candidates_by_token_overlap(
            tokens,
            domain_filter,
            min_overlap=1,
            limit=self.config.max_fuzzy_candidates
        )

        if not candidates:
            return []

        # Prepare for fuzzy matching
        candidate_texts = [c["search_text_normalized"] or c["search_text"] for c in candidates]

        # Use RapidFuzz process.extract for efficient batch matching
        # WRatio handles word reordering, which is common in medical terms
        matches = rapidfuzz["process"].extract(
            normalized_query,
            candidate_texts,
            scorer=rapidfuzz["fuzz"].WRatio,
            limit=min(50, len(candidates)),
            score_cutoff=self.config.fuzzy_min_threshold
        )

        # Map back to concept data
        results = []
        for match_text, score, idx in matches:
            results.append({
                "concept": candidates[idx],
                "score": score,
                "matched_text": match_text
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    def _semantic_match(
        self,
        query: str,
        domain_filter: Optional[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find semantic matches using FAISS.

        Returns concept IDs with similarity scores.
        """
        semantic_search = self._get_semantic_search()
        if not semantic_search:
            return []

        results = semantic_search.search(query, k=top_k)

        # Fetch concept details
        semantic_matches = []
        for concept_id, score in results:
            concept = self.db.get_concept_by_id(concept_id)
            if concept:
                # Apply domain filter if specified
                if domain_filter and concept["domain_id"] != domain_filter:
                    continue

                semantic_matches.append({
                    "concept": concept,
                    "score": score,
                    "matched_text": concept["concept_name"]
                })

        return semantic_matches

    def _merge_and_rank(
        self,
        fuzzy_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
        domain_filter: Optional[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Merge fuzzy and semantic results with combined scoring.

        Scoring combines:
        - Match score (fuzzy or semantic)
        - Vocabulary preference (domain-specific)
        - Source type preference (name > synonym)
        """
        # Create unified candidate list with concept_id as key
        candidates_by_id: Dict[int, Dict[str, Any]] = {}

        # Add fuzzy results
        for r in fuzzy_results:
            concept_id = r["concept"]["concept_id"]
            candidates_by_id[concept_id] = {
                "concept": r["concept"],
                "fuzzy_score": r["score"] / 100.0,  # Normalize to 0-1
                "semantic_score": 0.0,
                "match_type": MatchType.FUZZY_NAME if r["concept"].get("source_type") == "name" else MatchType.FUZZY_SYNONYM
            }

        # Add/merge semantic results
        for r in semantic_results:
            concept_id = r["concept"]["concept_id"]
            if concept_id in candidates_by_id:
                candidates_by_id[concept_id]["semantic_score"] = r["score"]
            else:
                candidates_by_id[concept_id] = {
                    "concept": r["concept"],
                    "fuzzy_score": 0.0,
                    "semantic_score": r["score"],
                    "match_type": MatchType.SEMANTIC
                }

        # Calculate combined scores
        for concept_id, data in candidates_by_id.items():
            # Weighted combination of fuzzy and semantic
            # Fuzzy gets higher weight as it's more precise for exact terminology
            base_score = max(
                data["fuzzy_score"] * 0.7 + data["semantic_score"] * 0.3,
                data["fuzzy_score"],
                data["semantic_score"]
            )

            # Apply vocabulary preference
            vocab_weight = self._get_vocab_weight(
                data["concept"]["vocabulary_id"],
                data["concept"]["domain_id"]
            )

            # Slight boost for name matches vs synonym matches
            source_boost = 1.0 if data["concept"].get("source_type") == "name" else 0.98

            # Slight boost for standard concepts
            standard_boost = 1.0 if data["concept"].get("standard_concept") == "S" else 0.95

            data["combined_score"] = base_score * vocab_weight * source_boost * standard_boost

        # Sort by combined score
        ranked = sorted(
            candidates_by_id.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )

        return ranked[:top_k * 2]  # Return more than needed for downstream filtering

    def _get_vocab_weight(self, vocabulary_id: str, domain_id: str) -> float:
        """Get vocabulary preference weight for a domain."""
        prefs = self.config.vocabulary_preferences.get(
            domain_id,
            self.config.vocabulary_preferences.get("default", {})
        )
        return prefs.get(vocabulary_id, self.config.default_vocab_weight)

    def _select_best_match(
        self,
        matches: List[Dict[str, Any]],
        domain_filter: Optional[str]
    ) -> Dict[str, Any]:
        """
        Select best match from exact match results.

        Applies vocabulary preference ranking for ties.
        """
        if len(matches) == 1:
            return matches[0]

        # Score each match by vocabulary preference
        scored = []
        for m in matches:
            weight = self._get_vocab_weight(m["vocabulary_id"], m["domain_id"])
            # Prefer standard concepts
            if m.get("standard_concept") == "S":
                weight *= 1.05
            # Prefer name matches over synonyms
            if m.get("source_type") == "name":
                weight *= 1.02
            scored.append((weight, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _create_match(
        self,
        concept_data: Dict[str, Any],
        confidence: float,
        match_type: MatchType,
        matched_text: str
    ) -> ConceptMatch:
        """
        Create a ConceptMatch with standard concept resolution.

        Resolves non-standard concepts to standard via "Maps To" relationship.
        """
        concept_id = concept_data["concept_id"]
        standard_concept = concept_data.get("standard_concept")

        # Determine resolution status
        if standard_concept == "S":
            resolution_status = ResolutionStatus.ALREADY_STANDARD
            standard_concept_id = None
            standard_concept_name = None
        else:
            # Look up Maps To relationship
            maps_to = self.db.get_maps_to(concept_id)
            if maps_to:
                resolution_status = ResolutionStatus.MAPPED_TO_STANDARD
                standard_concept_id = maps_to["standard_concept_id"]
                standard_concept_name = maps_to["standard_concept_name"]
            else:
                resolution_status = ResolutionStatus.NO_MAPPING_AVAILABLE
                standard_concept_id = None
                standard_concept_name = None

        return ConceptMatch(
            concept_id=concept_id,
            concept_name=concept_data["concept_name"],
            domain_id=concept_data["domain_id"],
            vocabulary_id=concept_data["vocabulary_id"],
            concept_class_id=concept_data.get("concept_class_id", ""),
            standard_concept=standard_concept,
            confidence=min(confidence, 1.0),  # Cap at 1.0
            match_type=match_type,
            matched_text=matched_text,
            resolution_status=resolution_status,
            standard_concept_id=standard_concept_id,
            standard_concept_name=standard_concept_name,
            concept_code=concept_data.get("concept_code")
        )


# Module-level convenience function
_searcher: Optional[ConceptSearcher] = None


def get_searcher(
    config: LookupConfig = DEFAULT_CONFIG,
    enable_semantic: bool = True
) -> ConceptSearcher:
    """Get or create the singleton searcher instance."""
    global _searcher
    if _searcher is None:
        _searcher = ConceptSearcher(config, enable_semantic=enable_semantic)
    return _searcher
