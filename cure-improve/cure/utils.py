"""
Utility functions for CURE.
"""

import torch
from typing import List, Optional
from PIL import Image


def set_seed(seed: int) -> torch.Generator:
    """
    Set random seed for reproducibility and return a generator.

    Args:
        seed: Random seed value

    Returns:
        PyTorch Generator with the specified seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def save_images(
    images: List[Image.Image],
    output_dir: str,
    prefix: str = "output",
    format: str = "png"
) -> List[str]:
    """
    Save a list of images to disk.

    Args:
        images: List of PIL Images
        output_dir: Directory to save images
        prefix: Filename prefix
        format: Image format (png, jpg, etc.)

    Returns:
        List of saved file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"{prefix}_{i}.{format}")
        img.save(path)
        paths.append(path)

    return paths


def create_image_grid(
    images: List[Image.Image],
    rows: int,
    cols: int,
    padding: int = 10
) -> Image.Image:
    """
    Create a grid of images.

    Args:
        images: List of PIL Images
        rows: Number of rows
        cols: Number of columns
        padding: Padding between images in pixels

    Returns:
        Combined grid image
    """
    if len(images) == 0:
        raise ValueError("No images provided")

    # Get dimensions from first image
    w, h = images[0].size

    # Create grid
    grid_w = cols * w + (cols - 1) * padding
    grid_h = rows * h + (rows - 1) * padding
    grid = Image.new('RGB', (grid_w, grid_h), color='white')

    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        row = idx // cols
        col = idx % cols
        x = col * (w + padding)
        y = row * (h + padding)
        grid.paste(img, (x, y))

    return grid


EMBEDDING_MODES = ("mean_masked", "token_flat", "mean_all", "eos_only")


def aggregate_embeddings(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str = "mean_masked",
) -> torch.Tensor:
    """
    Aggregate token-level embeddings into concept-level representations.

    Args:
        embeddings: [batch, seq_len, hidden_dim] from text encoder
        attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding
        mode: Aggregation strategy
            - "mean_masked": Mean over non-padding tokens (default, recommended)
            - "token_flat": Flatten all tokens including padding (legacy, for ablation)
            - "mean_all": Mean over all tokens including padding
            - "eos_only": Use only the end-of-sequence token per prompt

    Returns:
        Aggregated embeddings:
            mean_masked/mean_all/eos_only: [batch, hidden_dim]
            token_flat: [batch * seq_len, hidden_dim]
    """
    if mode not in EMBEDDING_MODES:
        raise ValueError(f"Unknown embedding_mode '{mode}'. Choose from {EMBEDDING_MODES}")

    batch_size, seq_len, hidden_dim = embeddings.shape

    if mode == "token_flat":
        return embeddings.reshape(batch_size * seq_len, hidden_dim)

    if mode == "mean_all":
        return embeddings.mean(dim=1)  # [batch, hidden_dim]

    if mode == "eos_only":
        # EOS is the last real token per sequence (last 1 in attention_mask)
        # For CLIP, EOS is at the position of the last non-pad token
        lengths = attention_mask.sum(dim=1).long()  # [batch]
        eos_indices = (lengths - 1).clamp(min=0)
        return embeddings[torch.arange(batch_size, device=embeddings.device), eos_indices]

    # mode == "mean_masked" (default)
    mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
    masked_sum = (embeddings * mask).sum(dim=1)  # [batch, hidden_dim]
    token_counts = mask.sum(dim=1).clamp(min=1)  # [batch, 1]
    return masked_sum / token_counts  # [batch, hidden_dim]


def get_default_forget_prompts(concept: str) -> List[str]:
    """
    Generate default forget prompts for common concepts.

    Provides synonym expansion for better concept coverage.

    Args:
        concept: Base concept to forget

    Returns:
        List of related prompts
    """
    concept_synonyms = {
        # Objects from Imagenette (Paper Table 4)
        "cassette player": [
            "cassette player", "cassette tape player", "tape player",
            "cassette deck", "cassette", "a cassette player",
            "music cassette player", "portable cassette player"
        ],
        "chain saw": [
            "chain saw", "chainsaw", "power saw", "chain saw tool",
            "a chain saw", "electric chain saw", "gas chain saw"
        ],
        "french horn": [
            "french horn", "horn", "brass horn", "musical horn",
            "a french horn", "orchestra horn"
        ],
        "golf ball": [
            "golf ball", "golf", "golf ball sport", "white golf ball",
            "a golf ball", "small golf ball"
        ],
        "car": [
            # Multiple variations as per paper methodology
            "car", "automobile", "vehicle", "sedan", "coupe",
            "a car", "the car", "cars", "motor car",
            "sports car", "racing car", "vintage car",
            "SUV", "truck", "van", "hatchback", "minivan",
            "convertible", "pickup truck", "jeep", "wagon",
            "crossover", "roadster", "muscle car", "classic car"
        ],
        "dog": [
            "dog", "puppy", "canine", "hound", "pup", "doggy",
            "a dog", "the dog", "dogs", "pet dog"
        ],
        "cat": [
            "cat", "kitten", "feline", "kitty", "tabby",
            "a cat", "the cat", "cats", "pet cat"
        ],
        "person": [
            "person", "human", "man", "woman", "people",
            "figure", "individual", "face", "portrait",
            "a person", "the person", "people", "person face"
        ],
        # NSFW (Paper uses alpha=5)
        "nudity": [
            "nudity", "naked", "nude", "unclothed", "bare skin",
            "explicit", "nsfw", "adult content", "pornographic",
            "sexual content", "naked person", "nude figure"
        ],
        # Celebrities (Paper tests identity removal - 20 variations per concept)
        "taylor swift": [
            "Taylor Swift",
            "portrait of Taylor Swift",
            "Taylor Swift face",
            "Taylor Swift singer",
            "Taylor Swift musician",
            "Taylor Swift celebrity",
            "Taylor Swift movie",
            "Taylor Swift photo",
            "Taylor Swift image",
            "Taylor Swift blonde",
            "Taylor Swift woman",
            "Taylor Swift concert",
            "Taylor Swift performance",
            "Taylor Swift red carpet",
            "Taylor Swift award",
            "Taylor Swift dress",
            "Taylor Swift smile",
            "Taylor Swift person",
            "Taylor Swift actress",
            "Taylor Swift looking at camera",
        ],
        "elon musk": [
            "Elon Musk",
            "portrait of Elon Musk",
            "Elon Musk face",
            "Elon Musk businessman",
            "Elon Musk entrepreneur",
            "Elon Musk celebrity",
            "Elon Musk photo",
            "Elon Musk image",
            "Elon Musk man",
            "Elon Musk suit",
            "Elon Musk Tesla",
            "Elon Musk SpaceX",
            "Elon Musk interview",
            "Elon Musk presentation",
            "Elon Musk speaking",
            "Elon Musk event",
            "Elon Musk tech",
            "Elon Musk person",
            "Elon Musk looking at camera",
            "Elon Musk portrait photo",
        ],
        "jennifer lawrence": [
            "Jennifer Lawrence",
            "portrait of Jennifer Lawrence",
            "Jennifer Lawrence face",
            "Jennifer Lawrence actress",
            "Jennifer Lawrence celebrity",
            "Jennifer Lawrence movie",
            "Jennifer Lawrence photo",
            "Jennifer Lawrence image",
            "Jennifer Lawrence woman",
            "Jennifer Lawrence brunette",
            "Jennifer Lawrence red carpet",
            "Jennifer Lawrence award",
            "Jennifer Lawrence dress",
            "Jennifer Lawrence smile",
            "Jennifer Lawrence film",
            "Jennifer Lawrence person",
            "Jennifer Lawrence looking",
            "Jennifer Lawrence pretty",
            "Jennifer Lawrence beautiful",
            "Jennifer Lawrence portrait photo",
        ],
        "emma stone": [
            "Emma Stone",
            "portrait of Emma Stone",
            "Emma Stone face",
            "Emma Stone actress",
            "Emma Stone celebrity",
            "Emma Stone movie",
            "Emma Stone photo",
            "Emma Stone image",
            "Emma Stone woman",
            "Emma Stone redhead",
            "Emma Stone red hair",
            "Emma Stone red carpet",
            "Emma Stone award",
            "Emma Stone dress",
            "Emma Stone smile",
            "Emma Stone film",
            "Emma Stone person",
            "Emma Stone looking",
            "Emma Stone beautiful",
            "Emma Stone portrait photo",
        ],
    }

    # Return synonyms if available, otherwise just the concept
    return concept_synonyms.get(concept.lower(), [concept])


def get_default_retain_prompts(concept: str) -> List[str]:
    """
    Get default retain prompts for common concepts.

    These are related but distinct concepts that should be preserved.

    Args:
        concept: Base concept being erased

    Returns:
        List of related concepts to preserve
    """
    related_concepts = {
        "car": ["bicycle", "motorcycle", "bus", "truck", "train", "road", "street"],
        "dog": ["cat", "bird", "animal", "pet"],
        "cat": ["dog", "bird", "animal", "pet"],
        "person": ["statue", "mannequin", "robot", "doll"],
    }

    return related_concepts.get(concept.lower(), [])
