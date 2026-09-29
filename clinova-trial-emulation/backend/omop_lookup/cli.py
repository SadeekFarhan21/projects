"""
Command-line interface for OMOP Concept Lookup.

Usage:
    # Initialize database (first time)
    python -m omop_lookup init

    # Single lookup
    python -m omop_lookup search "heart attack"
    python -m omop_lookup search "knee surgery" --domain Procedure

    # Batch lookup
    python -m omop_lookup batch queries.txt --output results.csv

    # Build semantic index (optional, slow)
    python -m omop_lookup build-index --domain Condition
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_init(args):
    """Initialize the database."""
    from . import initialize
    initialize(force_rebuild=args.force)


def cmd_search(args):
    """Search for a single term."""
    from . import lookup

    result = lookup(
        args.query,
        domain=args.domain,
        top_k=args.top_k,
        enable_semantic=not args.no_semantic
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\nQuery: {result.query}")
        print(f"Normalized: {result.normalized_query}")
        print(f"Success: {result.success}")
        print(f"Search time: {result.search_time_ms:.2f}ms")

        if result.best_match:
            m = result.best_match
            print(f"\n--- Best Match ---")
            print(f"  Concept ID: {m.concept_id}")
            print(f"  Name: {m.concept_name}")
            print(f"  Domain: {m.domain_id}")
            print(f"  Vocabulary: {m.vocabulary_id}")
            print(f"  Confidence: {m.confidence:.4f}")
            print(f"  Match Type: {m.match_type.value}")
            print(f"  Resolution: {m.resolution_status.value}")

            analytics_id = m.get_analytics_concept_id()
            if analytics_id:
                print(f"  Analytics ID: {analytics_id}")
                if m.standard_concept_name:
                    print(f"  Standard Name: {m.standard_concept_name}")

        if result.candidates:
            print(f"\n--- Other Candidates ({len(result.candidates)}) ---")
            for i, c in enumerate(result.candidates, 1):
                print(f"  {i}. {c.concept_name} ({c.vocabulary_id}) - {c.confidence:.3f}")

        if result.warnings:
            print(f"\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")


def cmd_batch(args):
    """Process batch queries."""
    from . import batch_lookup

    results = batch_lookup(
        args.input,
        output_path=args.output,
        domain=args.domain,
        enable_semantic=not args.no_semantic
    )

    print(f"\nBatch Lookup Complete")
    print(f"  Total queries: {results.total_queries}")
    print(f"  Successful: {results.successful_queries}")
    print(f"  Success rate: {results.successful_queries / max(results.total_queries, 1) * 100:.1f}%")
    print(f"  Total time: {results.total_time_ms:.2f}ms")
    print(f"  Avg time/query: {results.total_time_ms / max(results.total_queries, 1):.2f}ms")

    if args.output:
        print(f"  Results written to: {args.output}")


def cmd_build_index(args):
    """Build semantic search index."""
    from . import build_semantic_index
    build_semantic_index(domain_filter=args.domain)


def cmd_stats(args):
    """Show database statistics."""
    from . import get_stats

    stats = get_stats()
    print("\nDatabase Statistics:")
    for table, count in stats.items():
        print(f"  {table}: {count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="OMOP Concept Lookup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize database from CSV files")
    init_parser.add_argument("--force", action="store_true", help="Force rebuild even if exists")

    # search command
    search_parser = subparsers.add_parser("search", help="Search for a single term")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--domain", "-d", help="Filter by domain (Condition, Procedure, Drug, etc.)")
    search_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of candidates")
    search_parser.add_argument("--no-semantic", action="store_true", help="Disable semantic search")
    search_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # batch command
    batch_parser = subparsers.add_parser("batch", help="Process batch queries from file")
    batch_parser.add_argument("input", help="Input file (.txt or .csv)")
    batch_parser.add_argument("--output", "-o", help="Output CSV file")
    batch_parser.add_argument("--domain", "-d", help="Filter by domain")
    batch_parser.add_argument("--no-semantic", action="store_true", help="Disable semantic search")

    # build-index command
    index_parser = subparsers.add_parser("build-index", help="Build semantic search FAISS index")
    index_parser.add_argument("--domain", "-d", help="Limit to specific domain")

    # stats command
    subparsers.add_parser("stats", help="Show database statistics")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "build-index":
        cmd_build_index(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
