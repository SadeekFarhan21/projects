# OMOP Concept Lookup Tool

A high-accuracy, low-latency tool for mapping free-text medical terms to OMOP/Athena concept IDs.

## Features

- **Hybrid Search**: Exact match → Fuzzy match → Semantic search (tiered with early exit)
- **Standard Concept Resolution**: Automatically maps non-standard concepts to standard via "Maps To"
- **Domain-Aware Ranking**: Vocabulary preferences based on domain (Condition, Procedure, Drug, etc.)
- **Agent-Friendly**: JSON-in/JSON-out interface for multi-agent systems
- **Fully Local**: No external APIs required

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize Database

Place OMOP vocabulary files in the `omop_lookup/` directory:
- `CONCEPT.csv`
- `CONCEPT_SYNONYM.csv`
- `CONCEPT_RELATIONSHIP.csv`

Then initialize:

```bash
python -m omop_lookup init
```

This creates a DuckDB database with optimized tables for fast lookup.

### 3. Search for Concepts

**Command Line:**
```bash
# Single search
python -m omop_lookup search "heart attack" --domain Condition

# Batch search
python -m omop_lookup batch queries.txt --output results.csv
```

**Python:**
```python
from omop_lookup import lookup

result = lookup("heart attack", domain="Condition")

if result.success:
    match = result.best_match
    print(f"Concept ID: {match.concept_id}")
    print(f"Name: {match.concept_name}")
    print(f"Analytics ID: {match.get_analytics_concept_id()}")
```

## Agent Integration

For multi-agent systems, use the agent interface:

```python
from omop_lookup.agent_interface import lookup_concept

# Returns JSON-serializable dict
result = lookup_concept("myocardial infarction", domain="Condition")

if result["success"]:
    concept_id = result["analytics_concept_id"]  # Use this for OMOP analytics
```

## Semantic Search (Optional)

For better matching of synonyms and related terms, build the FAISS index:

```bash
# Build for specific domain (faster)
python -m omop_lookup build-index --domain Condition

# Build for all domains (slow, hours)
python -m omop_lookup build-index
```

Requires: `torch`, `transformers`, `faiss-cpu`

Uses SapBERT, a model trained specifically for medical concept linking.

## Architecture

```
Query
  │
  ▼
┌─────────────────┐
│  Normalize Text │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────┐
│  Exact Match    │────►│   Return    │  (confidence = 1.0)
└────────┬────────┘     └─────────────┘
         │ (no match)
         ▼
┌─────────────────┐
│  Token Filter   │  (reduce to ~1000 candidates)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────┐
│  Fuzzy Match    │────►│   Return    │  (if score > 0.95)
└────────┬────────┘     └─────────────┘
         │ (low confidence)
         ▼
┌─────────────────┐
│ Semantic Search │  (FAISS nearest neighbors)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Merge & Rank   │  (combine scores, apply vocab preferences)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Resolve to Std  │  (Maps To lookup)
└────────┬────────┘
         │
         ▼
    Return Result
```

## Configuration

Edit `config.py` to adjust:

- Search thresholds
- Vocabulary preferences
- FAISS parameters
- Embedding model

## Output Format

```json
{
  "success": true,
  "query": "heart attack",
  "best_match": {
    "concept_id": 4329847,
    "concept_name": "Myocardial infarction",
    "domain_id": "Condition",
    "vocabulary_id": "SNOMED",
    "confidence": 0.9523,
    "match_type": "fuzzy_name",
    "resolution_status": "already_standard"
  },
  "analytics_concept_id": 4329847,
  "candidates": [...],
  "warnings": []
}
```

## Key Fields

- **concept_id**: Matched concept ID
- **analytics_concept_id**: ID to use for OMOP analytics (resolved to standard)
- **confidence**: Match confidence (0-1)
- **resolution_status**:
  - `already_standard`: Concept is standard
  - `mapped_to_standard`: Resolved via "Maps To"
  - `no_mapping_available`: No standard mapping exists
