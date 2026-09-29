"""
Text normalization utilities for medical concept matching.

Medical-specific considerations:
- Preserve clinically significant punctuation (e.g., "Type 2" vs "Type II")
- Handle common medical abbreviations
- Normalize whitespace and case consistently
"""

import re
import unicodedata
from typing import Optional


def normalize_text(text: str) -> str:
    """
    Normalize text for concept matching.

    Steps:
    1. Unicode normalization (NFKD form)
    2. Lowercase conversion
    3. Remove accents/diacritics
    4. Normalize whitespace
    5. Remove non-alphanumeric characters (except spaces)

    Args:
        text: Raw input text

    Returns:
        Normalized text for matching

    Example:
        >>> normalize_text("Myocardial Infarction (Heart Attack)")
        'myocardial infarction heart attack'
    """
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Lowercase
    text = text.lower()

    # Remove accents/diacritics (keep base characters)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    # Replace common punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace (multiple spaces -> single, strip edges)
    text = " ".join(text.split())

    return text


def normalize_for_exact_match(text: str) -> str:
    """
    Stricter normalization for exact matching.

    Same as normalize_text but also removes numbers for matching
    concepts that may differ only in numbering conventions.

    Example:
        >>> normalize_for_exact_match("Type 2 Diabetes")
        'type diabetes'
    """
    normalized = normalize_text(text)
    # Optionally remove standalone numbers (not part of words)
    # This is commented out as it may be too aggressive for some use cases
    # normalized = re.sub(r'\b\d+\b', '', normalized)
    # normalized = ' '.join(normalized.split())
    return normalized


def tokenize(text: str) -> list:
    """
    Tokenize normalized text into words.

    Args:
        text: Normalized text

    Returns:
        List of tokens

    Example:
        >>> tokenize("myocardial infarction")
        ['myocardial', 'infarction']
    """
    return text.split()


def get_token_set(text: str) -> set:
    """
    Get set of unique tokens from text.

    Useful for token-overlap filtering before fuzzy matching.

    Example:
        >>> get_token_set("acute myocardial infarction")
        {'acute', 'myocardial', 'infarction'}
    """
    return set(tokenize(normalize_text(text)))


def calculate_token_overlap(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between token sets.

    Used for fast pre-filtering before expensive fuzzy/semantic matching.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity (0.0 to 1.0)

    Example:
        >>> calculate_token_overlap("heart attack", "myocardial infarction heart")
        0.25  # 1 shared token / 4 unique tokens
    """
    tokens1 = get_token_set(text1)
    tokens2 = get_token_set(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)


def expand_abbreviations(text: str) -> str:
    """
    Expand common medical abbreviations.

    Note: This is a basic implementation. Production systems should use
    a comprehensive medical abbreviation dictionary.

    Args:
        text: Input text (already normalized)

    Returns:
        Text with abbreviations expanded
    """
    # Common medical abbreviations (lowercase, normalized)
    abbreviations = {
        "mi": "myocardial infarction",
        "dm": "diabetes mellitus",
        "htn": "hypertension",
        "chf": "congestive heart failure",
        "copd": "chronic obstructive pulmonary disease",
        "cad": "coronary artery disease",
        "afib": "atrial fibrillation",
        "ckd": "chronic kidney disease",
        "uti": "urinary tract infection",
        "dvt": "deep vein thrombosis",
        "pe": "pulmonary embolism",
        "ra": "rheumatoid arthritis",
        "oa": "osteoarthritis",
        "ms": "multiple sclerosis",
        "als": "amyotrophic lateral sclerosis",
        "tia": "transient ischemic attack",
        "cva": "cerebrovascular accident",
        "gerd": "gastroesophageal reflux disease",
        "ibs": "irritable bowel syndrome",
        "ards": "acute respiratory distress syndrome",
        "aki": "acute kidney injury",
        "esrd": "end stage renal disease",
        "bph": "benign prostatic hyperplasia",
        "pci": "percutaneous coronary intervention",
        "cabg": "coronary artery bypass graft",
        "tka": "total knee arthroplasty",
        "tha": "total hip arthroplasty",
        "acl": "anterior cruciate ligament",
        "mri": "magnetic resonance imaging",
        "ct": "computed tomography",
        "ekg": "electrocardiogram",
        "ecg": "electrocardiogram",
        "cbc": "complete blood count",
        "bmp": "basic metabolic panel",
        "cmp": "comprehensive metabolic panel",
        "lfts": "liver function tests",
        "hba1c": "hemoglobin a1c",
        "bp": "blood pressure",
        "hr": "heart rate",
        "rr": "respiratory rate",
        "o2sat": "oxygen saturation",
    }

    normalized = normalize_text(text)
    tokens = tokenize(normalized)

    # Check if the entire normalized text is an abbreviation
    if normalized in abbreviations:
        return abbreviations[normalized]

    # Check individual tokens
    expanded_tokens = []
    for token in tokens:
        if token in abbreviations:
            expanded_tokens.append(abbreviations[token])
        else:
            expanded_tokens.append(token)

    return " ".join(expanded_tokens)


def generate_search_variants(text: str) -> list:
    """
    Generate search variants for a query.

    Returns multiple normalized forms to try for matching:
    1. Basic normalized form
    2. With abbreviations expanded
    3. With common typo corrections (future enhancement)

    Args:
        text: Raw input text

    Returns:
        List of search variants (deduplicated)
    """
    variants = set()

    # Basic normalization
    basic = normalize_text(text)
    variants.add(basic)

    # With abbreviation expansion
    expanded = expand_abbreviations(text)
    if expanded != basic:
        variants.add(expanded)

    return list(variants)
