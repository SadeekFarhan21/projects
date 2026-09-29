"""
DuckDB database layer for OMOP vocabulary data.

Handles:
- Loading CSV files into optimized tables
- Creating denormalized lookup tables (Maps To)
- Indexed queries for exact matching
"""

import duckdb
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .config import LookupConfig, DEFAULT_CONFIG
from .normalize import normalize_text

logger = logging.getLogger(__name__)


class OMOPDatabase:
    """
    DuckDB-based storage for OMOP vocabulary data.

    Creates optimized tables:
    - concept: Main concept table with normalized names
    - concept_synonym: Synonyms with normalized text
    - maps_to_lookup: Denormalized source->standard mappings
    - concept_search: Combined view for efficient searching
    """

    def __init__(self, config: LookupConfig = DEFAULT_CONFIG):
        self.config = config
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self._initialized = False

    def connect(self) -> None:
        """Connect to or create the DuckDB database."""
        self.conn = duckdb.connect(str(self.config.db_path))
        logger.info(f"Connected to database: {self.config.db_path}")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def is_initialized(self) -> bool:
        """Check if the database has been initialized with data."""
        if not self.conn:
            self.connect()

        try:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'concept'"
            ).fetchone()
            return result[0] > 0
        except Exception:
            return False

    def initialize(self, force_rebuild: bool = False) -> None:
        """
        Initialize the database by loading CSV files and creating lookup tables.

        Args:
            force_rebuild: If True, drop existing tables and reload
        """
        if not self.conn:
            self.connect()

        if self.is_initialized() and not force_rebuild:
            logger.info("Database already initialized. Use force_rebuild=True to reload.")
            self._initialized = True
            return

        logger.info("Initializing OMOP database...")

        # Drop existing tables if rebuilding
        if force_rebuild:
            self._drop_tables()

        # Load source CSV files
        self._load_concept_table()
        self._load_synonym_table()
        self._load_relationship_table()

        # Create denormalized lookup tables
        self._create_maps_to_lookup()
        self._create_search_table()

        # Create indices for fast lookups
        self._create_indices()

        self._initialized = True
        logger.info("Database initialization complete.")

    def _drop_tables(self) -> None:
        """Drop all existing tables."""
        tables = [
            "concept_search",
            "maps_to_lookup",
            "concept_relationship",
            "concept_synonym",
            "concept"
        ]
        for table in tables:
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        logger.info("Dropped existing tables.")

    def _load_concept_table(self) -> None:
        """Load CONCEPT.csv into DuckDB with normalized names."""
        logger.info(f"Loading concepts from {self.config.concept_path}...")

        # Load raw CSV
        self.conn.execute(f"""
            CREATE TABLE concept AS
            SELECT
                CAST(concept_id AS INTEGER) as concept_id,
                concept_name,
                domain_id,
                vocabulary_id,
                concept_class_id,
                standard_concept,
                concept_code,
                valid_start_date,
                valid_end_date,
                invalid_reason
            FROM read_csv_auto('{self.config.concept_path}',
                               delim='\t',
                               header=true,
                               ignore_errors=true)
        """)

        # Add normalized name column
        self.conn.execute("ALTER TABLE concept ADD COLUMN name_normalized VARCHAR")

        # Count rows
        count = self.conn.execute("SELECT COUNT(*) FROM concept").fetchone()[0]
        logger.info(f"Loaded {count:,} concepts.")

    def _load_synonym_table(self) -> None:
        """Load CONCEPT_SYNONYM.csv into DuckDB."""
        logger.info(f"Loading synonyms from {self.config.synonym_path}...")

        self.conn.execute(f"""
            CREATE TABLE concept_synonym AS
            SELECT
                CAST(concept_id AS INTEGER) as concept_id,
                concept_synonym_name,
                CAST(language_concept_id AS INTEGER) as language_concept_id
            FROM read_csv_auto('{self.config.synonym_path}',
                               delim='\t',
                               header=true,
                               ignore_errors=true)
        """)

        # Add normalized synonym column
        self.conn.execute("ALTER TABLE concept_synonym ADD COLUMN synonym_normalized VARCHAR")

        count = self.conn.execute("SELECT COUNT(*) FROM concept_synonym").fetchone()[0]
        logger.info(f"Loaded {count:,} synonyms.")

    def _load_relationship_table(self) -> None:
        """Load CONCEPT_RELATIONSHIP.csv into DuckDB."""
        logger.info(f"Loading relationships from {self.config.relationship_path}...")

        self.conn.execute(f"""
            CREATE TABLE concept_relationship AS
            SELECT
                CAST(concept_id_1 AS INTEGER) as concept_id_1,
                CAST(concept_id_2 AS INTEGER) as concept_id_2,
                relationship_id,
                valid_start_date,
                valid_end_date,
                invalid_reason
            FROM read_csv_auto('{self.config.relationship_path}',
                               delim='\t',
                               header=true,
                               ignore_errors=true)
        """)

        count = self.conn.execute("SELECT COUNT(*) FROM concept_relationship").fetchone()[0]
        logger.info(f"Loaded {count:,} relationships.")

    def _create_maps_to_lookup(self) -> None:
        """
        Create denormalized Maps To lookup table.

        This is critical for OMOP compliance: maps non-standard concepts
        to their standard equivalents for analytics.

        The "Maps to" relationship connects:
        - Source vocabulary concepts (ICD-10, etc.) -> Standard concepts (SNOMED)
        """
        logger.info("Creating Maps To lookup table...")

        self.conn.execute("""
            CREATE TABLE maps_to_lookup AS
            SELECT DISTINCT
                cr.concept_id_1 AS source_concept_id,
                target.concept_id AS standard_concept_id,
                target.concept_name AS standard_concept_name,
                target.domain_id AS standard_domain_id,
                target.vocabulary_id AS standard_vocabulary_id
            FROM concept_relationship cr
            JOIN concept target ON cr.concept_id_2 = target.concept_id
            WHERE cr.relationship_id = 'Maps to'
              AND cr.invalid_reason IS NULL
              AND target.standard_concept = 'S'
              AND target.invalid_reason IS NULL
        """)

        count = self.conn.execute("SELECT COUNT(*) FROM maps_to_lookup").fetchone()[0]
        logger.info(f"Created Maps To lookup with {count:,} mappings.")

    def _create_search_table(self) -> None:
        """
        Create unified search table combining concepts and synonyms.

        This table has:
        - All concept names (from concept table)
        - All synonyms (from concept_synonym table)
        - Normalized text for matching (done in SQL for speed)
        - Pre-joined with concept metadata
        """
        logger.info("Creating unified search table...")

        # First create the table without normalization
        self.conn.execute("""
            CREATE TABLE concept_search AS

            -- Concept names
            SELECT
                c.concept_id,
                c.concept_name AS search_text,
                'name' AS source_type,
                c.concept_name,
                c.domain_id,
                c.vocabulary_id,
                c.concept_class_id,
                c.standard_concept,
                c.concept_code,
                c.invalid_reason
            FROM concept c
            WHERE c.invalid_reason IS NULL

            UNION ALL

            -- Synonyms (joined with concept metadata)
            SELECT
                s.concept_id,
                s.concept_synonym_name AS search_text,
                'synonym' AS source_type,
                c.concept_name,
                c.domain_id,
                c.vocabulary_id,
                c.concept_class_id,
                c.standard_concept,
                c.concept_code,
                c.invalid_reason
            FROM concept_synonym s
            JOIN concept c ON s.concept_id = c.concept_id
            WHERE c.invalid_reason IS NULL
        """)

        count = self.conn.execute("SELECT COUNT(*) FROM concept_search").fetchone()[0]
        logger.info(f"Created search table with {count:,} entries.")

        # Add normalized column and populate using simple SQL functions
        # This is more robust than complex regex
        logger.info("Adding normalized text column...")
        self.conn.execute("ALTER TABLE concept_search ADD COLUMN search_text_normalized VARCHAR")

        logger.info("Populating normalized text (this may take a few minutes)...")
        # Use simple string functions that won't fail on Unicode:
        # 1. lower() - lowercase
        # 2. Replace common punctuation with spaces
        # 3. Use regexp_replace only for collapsing multiple spaces
        self.conn.execute("""
            UPDATE concept_search
            SET search_text_normalized = trim(
                regexp_replace(
                    translate(
                        lower(COALESCE(search_text, '')),
                        '.,;:!?()[]{}/<>\\|@#$%^&*+=~`"''_-',
                        '                                   '
                    ),
                    ' +', ' ', 'g'
                )
            )
        """)

        # Verify normalization
        populated = self.conn.execute("""
            SELECT COUNT(*) FROM concept_search
            WHERE search_text_normalized IS NOT NULL AND search_text_normalized != ''
        """).fetchone()[0]
        logger.info(f"Normalized {populated:,} of {count:,} entries.")

    def _create_indices(self) -> None:
        """Create indices for fast lookups."""
        logger.info("Creating indices...")

        # Index on concept_id for joins
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_id ON concept(concept_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_search_concept_id ON concept_search(concept_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_maps_source ON maps_to_lookup(source_concept_id)")

        # Index on domain for filtering
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_search_domain ON concept_search(domain_id)")

        logger.info("Indices created.")

    def update_normalized_text(self) -> None:
        """
        Update normalized text columns.

        Note: Normalization is now done inline during table creation via SQL,
        so this method is kept for backwards compatibility but does nothing.
        """
        logger.info("Normalization already done during table creation (SQL-based).")

    def exact_match(
        self,
        normalized_query: str,
        domain_filter: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find exact matches on normalized text.

        Args:
            normalized_query: Pre-normalized search text
            domain_filter: Optional domain to filter (e.g., 'Condition', 'Procedure')
            limit: Maximum results to return

        Returns:
            List of matching concept dictionaries
        """
        if not self.conn:
            self.connect()

        query = """
            SELECT
                concept_id,
                search_text,
                source_type,
                concept_name,
                domain_id,
                vocabulary_id,
                concept_class_id,
                standard_concept,
                concept_code
            FROM concept_search
            WHERE search_text_normalized = ?
        """
        params = [normalized_query]

        if domain_filter:
            query += " AND domain_id = ?"
            params.append(domain_filter)

        query += f" LIMIT {limit}"

        results = self.conn.execute(query, params).fetchall()
        columns = [
            "concept_id", "search_text", "source_type", "concept_name",
            "domain_id", "vocabulary_id", "concept_class_id",
            "standard_concept", "concept_code"
        ]

        return [dict(zip(columns, row)) for row in results]

    def get_candidates_by_token_overlap(
        self,
        tokens: List[str],
        domain_filter: Optional[str] = None,
        min_overlap: int = 1,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get candidate concepts that share tokens with the query.

        This is used for pre-filtering before expensive fuzzy matching.

        Args:
            tokens: List of normalized tokens from query
            domain_filter: Optional domain filter
            min_overlap: Minimum number of shared tokens
            limit: Maximum candidates to return

        Returns:
            List of candidate concept dictionaries
        """
        if not self.conn or not tokens:
            return []

        # Build query with token matching
        # Uses LIKE for each token to find candidates
        token_conditions = " OR ".join(
            f"search_text_normalized LIKE '%{token}%'" for token in tokens
        )

        query = f"""
            SELECT
                concept_id,
                search_text,
                search_text_normalized,
                source_type,
                concept_name,
                domain_id,
                vocabulary_id,
                concept_class_id,
                standard_concept,
                concept_code
            FROM concept_search
            WHERE ({token_conditions})
        """

        if domain_filter:
            query += f" AND domain_id = '{domain_filter}'"

        query += f" LIMIT {limit}"

        results = self.conn.execute(query).fetchall()
        columns = [
            "concept_id", "search_text", "search_text_normalized", "source_type",
            "concept_name", "domain_id", "vocabulary_id", "concept_class_id",
            "standard_concept", "concept_code"
        ]

        return [dict(zip(columns, row)) for row in results]

    def get_maps_to(self, concept_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the standard concept that this concept maps to.

        Args:
            concept_id: Source concept ID

        Returns:
            Standard concept info dict or None if no mapping exists
        """
        if not self.conn:
            self.connect()

        result = self.conn.execute("""
            SELECT
                standard_concept_id,
                standard_concept_name,
                standard_domain_id,
                standard_vocabulary_id
            FROM maps_to_lookup
            WHERE source_concept_id = ?
            LIMIT 1
        """, [concept_id]).fetchone()

        if result:
            return {
                "standard_concept_id": result[0],
                "standard_concept_name": result[1],
                "standard_domain_id": result[2],
                "standard_vocabulary_id": result[3],
            }
        return None

    def get_concept_by_id(self, concept_id: int) -> Optional[Dict[str, Any]]:
        """
        Get full concept details by ID.

        Args:
            concept_id: Concept ID to look up

        Returns:
            Concept dictionary or None if not found
        """
        if not self.conn:
            self.connect()

        result = self.conn.execute("""
            SELECT
                concept_id,
                concept_name,
                domain_id,
                vocabulary_id,
                concept_class_id,
                standard_concept,
                concept_code
            FROM concept
            WHERE concept_id = ?
        """, [concept_id]).fetchone()

        if result:
            return {
                "concept_id": result[0],
                "concept_name": result[1],
                "domain_id": result[2],
                "vocabulary_id": result[3],
                "concept_class_id": result[4],
                "standard_concept": result[5],
                "concept_code": result[6],
            }
        return None

    def get_all_concepts_for_embedding(
        self,
        batch_size: int = 10000,
        domain_filter: Optional[str] = None
    ):
        """
        Generator yielding all concepts for embedding generation.

        Yields batches of (concept_id, search_text) tuples.

        Args:
            batch_size: Number of records per batch
            domain_filter: Optional domain filter

        Yields:
            List of (concept_id, search_text, search_text_normalized) tuples
        """
        if not self.conn:
            self.connect()

        query = """
            SELECT concept_id, search_text, search_text_normalized
            FROM concept_search
        """
        if domain_filter:
            query += f" WHERE domain_id = '{domain_filter}'"

        # Use streaming query
        result = self.conn.execute(query)

        while True:
            batch = result.fetchmany(batch_size)
            if not batch:
                break
            yield batch

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        if not self.conn:
            self.connect()

        stats = {}

        tables = ["concept", "concept_synonym", "concept_relationship",
                  "maps_to_lookup", "concept_search"]

        for table in tables:
            try:
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = count
            except Exception:
                stats[table] = 0

        return stats


# Module-level convenience functions
_db_instance: Optional[OMOPDatabase] = None


def get_database(config: LookupConfig = DEFAULT_CONFIG) -> OMOPDatabase:
    """Get or create the singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = OMOPDatabase(config)
        _db_instance.connect()
    return _db_instance
