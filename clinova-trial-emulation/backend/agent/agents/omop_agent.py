"""
OMOP Lookup Agent

Resolves medical terms to OMOP concept IDs using the omop_lookup tool.
Takes a list of terms from the design spec and returns top-10 matches for each.
"""

import csv
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for omop_lookup import
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class OMOPLookupAgent:
    """
    Agent that resolves medical terms to OMOP concept IDs.

    Uses the omop_lookup tool to perform hybrid search (exact → fuzzy → semantic)
    and returns the top matches for each term.
    """

    def __init__(self, top_k: int = 10, enable_semantic: bool = False):
        """
        Initialize the OMOP Lookup Agent.

        Args:
            top_k: Number of top matches to return per term (default: 10)
            enable_semantic: Whether to use semantic search (requires built index)
        """
        self.top_k = top_k
        self.enable_semantic = enable_semantic
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Ensure the OMOP lookup database is initialized."""
        if self._initialized:
            return True

        try:
            from omop_lookup import get_stats
            stats = get_stats()
            if stats.get("concept", 0) > 0:
                self._initialized = True
                return True
            else:
                print("Warning: OMOP database appears empty. Run 'python -m omop_lookup init' first.")
                return False
        except Exception as e:
            print(f"Warning: OMOP lookup not available: {e}")
            return False

    def lookup_terms(
        self,
        terms: list[tuple[str, Optional[str]]],
    ) -> dict[str, list[dict]]:
        """
        Look up multiple terms and return top matches for each.

        Args:
            terms: List of (term, domain) tuples where domain is optional

        Returns:
            Dictionary mapping each term to its list of matches.
            Each match contains: concept_id, concept_name, domain_id,
            vocabulary_id, confidence, analytics_concept_id
        """
        if not self._ensure_initialized():
            # Return empty results with warning
            return {term: [] for term, _ in terms}

        from omop_lookup import lookup

        results = {}

        for term, domain in terms:
            try:
                result = lookup(
                    term,
                    domain=domain,
                    top_k=self.top_k,
                    enable_semantic=self.enable_semantic,
                )

                matches = []

                # Add best match if exists
                if result.best_match:
                    m = result.best_match
                    matches.append({
                        "concept_id": m.concept_id,
                        "concept_name": m.concept_name,
                        "domain_id": m.domain_id,
                        "vocabulary_id": m.vocabulary_id,
                        "confidence": round(m.confidence, 4),
                        "match_type": m.match_type.value,
                        "analytics_concept_id": m.get_analytics_concept_id(),
                        "standard_concept_name": m.standard_concept_name,
                    })

                # Add other candidates
                for c in result.candidates:
                    matches.append({
                        "concept_id": c.concept_id,
                        "concept_name": c.concept_name,
                        "domain_id": c.domain_id,
                        "vocabulary_id": c.vocabulary_id,
                        "confidence": round(c.confidence, 4),
                        "match_type": c.match_type.value,
                        "analytics_concept_id": c.get_analytics_concept_id(),
                        "standard_concept_name": c.standard_concept_name,
                    })

                results[term] = matches[:self.top_k]

            except Exception as e:
                print(f"Warning: Failed to lookup '{term}': {e}")
                results[term] = []

        return results

    def format_for_code_agent(self, lookup_results: dict[str, list[dict]]) -> str:
        """
        Format lookup results as context for the code agent.

        Args:
            lookup_results: Dictionary from lookup_terms()

        Returns:
            Formatted string with OMOP concept mappings
        """
        lines = ["# OMOP Concept ID Mappings", ""]
        lines.append("The following medical terms have been resolved to OMOP concept IDs.")
        lines.append("Use the `analytics_concept_id` for queries (resolved to standard concept).")
        lines.append("")

        for term, matches in lookup_results.items():
            lines.append(f"## {term}")

            if not matches:
                lines.append("  No matches found - use placeholder")
                lines.append("")
                continue

            # Show top match prominently
            top = matches[0]
            lines.append(f"  **Best Match:** {top['concept_name']}")
            lines.append(f"  - Concept ID: {top['concept_id']}")
            lines.append(f"  - Analytics ID: {top['analytics_concept_id']} (use this)")
            lines.append(f"  - Domain: {top['domain_id']}")
            lines.append(f"  - Vocabulary: {top['vocabulary_id']}")
            lines.append(f"  - Confidence: {top['confidence']}")

            # Show alternatives if available
            if len(matches) > 1:
                lines.append("")
                lines.append("  Alternatives:")
                for m in matches[1:5]:  # Show up to 4 alternatives
                    lines.append(
                        f"  - {m['concept_name']} (ID: {m['analytics_concept_id']}, "
                        f"conf: {m['confidence']})"
                    )

            lines.append("")

        return "\n".join(lines)

    def save_lookup_terms_file(
        self,
        terms: list[tuple[str, Optional[str]]],
        output_path: str,
    ) -> str:
        """
        Save terms to a lookup file for batch processing.

        Args:
            terms: List of (term, domain) tuples
            output_path: Path to save the file

        Returns:
            Path to the saved file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for term, domain in terms:
                if domain:
                    f.write(f"{term}|{domain}\n")
                else:
                    f.write(f"{term}\n")
        return output_path

    def save_results_csv(
        self,
        lookup_results: dict[str, list[dict]],
        output_path: str,
    ) -> str:
        """
        Save lookup results to a CSV file.

        Args:
            lookup_results: Dictionary from lookup_terms()
            output_path: Path to save the CSV

        Returns:
            Path to the saved file
        """
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_term",
                "rank",
                "concept_id",
                "concept_name",
                "analytics_concept_id",
                "domain_id",
                "vocabulary_id",
                "confidence",
                "match_type",
            ])

            for term, matches in lookup_results.items():
                if not matches:
                    writer.writerow([term, 1, "", "NO_MATCH", "", "", "", 0, "none"])
                else:
                    for rank, m in enumerate(matches, 1):
                        writer.writerow([
                            term,
                            rank,
                            m["concept_id"],
                            m["concept_name"],
                            m["analytics_concept_id"],
                            m["domain_id"],
                            m["vocabulary_id"],
                            m["confidence"],
                            m["match_type"],
                        ])

        return output_path
