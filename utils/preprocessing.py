from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Card text preprocessing
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")

# Common Hearthstone keyword tokens to preserve
HEARTHSTONE_KEYWORDS = frozenset(
    [
        "battlecry",
        "deathrattle",
        "discover",
        "divine shield",
        "echo",
        "freeze",
        "lifesteal",
        "magnetic",
        "overkill",
        "overload",
        "poisonous",
        "reborn",
        "rush",
        "silence",
        "spell damage",
        "stealth",
        "taunt",
        "tradeable",
        "windfury",
    ]
)


def clean_card_text(text: str) -> str:
    """Remove HTML tags and normalise whitespace in card description text.

    Args:
        text: Raw card description string (may contain HTML markup).

    Returns:
        Cleaned plain-text string.
    """
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def normalise_card_stats(card: dict[str, Any]) -> dict[str, float]:
    """Extract and normalise numeric card statistics.

    Performs min-max normalisation assuming standard Hearthstone ranges:
    - Cost: [0, 10]
    - Attack / Health: [0, 12]

    Args:
        card: Raw card metadata dictionary.

    Returns:
        Dictionary of normalised float statistics.
    """
    def _clamp_norm(value: int | float, lo: float, hi: float) -> float:
        return max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))

    stats: dict[str, float] = {}
    if "cost" in card:
        stats["cost"] = _clamp_norm(card["cost"], 0, 10)
    if "attack" in card:
        stats["attack"] = _clamp_norm(card["attack"], 0, 12)
    if "health" in card:
        stats["health"] = _clamp_norm(card["health"], 0, 12)
    return stats


def build_card_feature_vector(card: dict[str, Any], text_embedding: torch.Tensor) -> torch.Tensor:
    """Concatenate numeric card stats with a pre-computed text embedding.

    Args:
        card: Raw card metadata dictionary.
        text_embedding: 1-D tensor of shape (embedding_dim,) from the CLIP
            text encoder.

    Returns:
        Combined feature vector of shape (embedding_dim + num_stats,).
    """
    stats = normalise_card_stats(card)
    stat_tensor = torch.tensor(list(stats.values()), dtype=torch.float32)
    return torch.cat([text_embedding, stat_tensor], dim=-1)


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

CARD_TYPE_MAP: dict[str, int] = {
    "MINION": 0,
    "SPELL": 1,
    "WEAPON": 2,
    "HERO": 3,
    "HERO_POWER": 4,
    "LOCATION": 5,
}

CLASS_MAP: dict[str, int] = {
    "NEUTRAL": 0,
    "DRUID": 1,
    "HUNTER": 2,
    "MAGE": 3,
    "PALADIN": 4,
    "PRIEST": 5,
    "ROGUE": 6,
    "SHAMAN": 7,
    "WARLOCK": 8,
    "WARRIOR": 9,
    "DEMON_HUNTER": 10,
    "DEATH_KNIGHT": 11,
}


def card_to_one_hot(card: dict[str, Any]) -> torch.Tensor:
    """Encode card type and class as a concatenated one-hot vector.

    Args:
        card: Raw card metadata dictionary.

    Returns:
        One-hot tensor of shape (len(CARD_TYPE_MAP) + len(CLASS_MAP),).
    """
    type_vec = torch.zeros(len(CARD_TYPE_MAP))
    class_vec = torch.zeros(len(CLASS_MAP))

    card_type = card.get("type", "").upper()
    if card_type in CARD_TYPE_MAP:
        type_vec[CARD_TYPE_MAP[card_type]] = 1.0

    card_class = card.get("cardClass", "NEUTRAL").upper()
    if card_class in CLASS_MAP:
        class_vec[CLASS_MAP[card_class]] = 1.0

    return torch.cat([type_vec, class_vec], dim=0)


def preprocess_card_data(
    cards: list[dict[str, Any]],
    text_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Preprocess a list of raw card dicts into a feature matrix.

    For each card, the following features are concatenated:
    1. Normalised numeric stats (cost, attack, health).
    2. One-hot card type + class encoding.
    3. CLIP text embedding.

    Args:
        cards: List of raw card metadata dicts.
        text_embeddings: Tensor of shape (N, embedding_dim) produced by
            :class:`~models.clip_encoder.CLIPEncoder`.

    Returns:
        Tuple of:
            - feature_matrix: Float tensor of shape (N, feature_dim).
            - cleaned_cards: List of card dicts with cleaned text fields.
    """
    feature_rows: list[torch.Tensor] = []
    cleaned_cards: list[dict[str, Any]] = []

    for card, emb in zip(cards, text_embeddings):
        # Clean text in-place copy
        cleaned = dict(card)
        if "text" in cleaned:
            cleaned["text"] = clean_card_text(cleaned["text"])

        # Build feature
        stat_tensor = torch.tensor(
            list(normalise_card_stats(cleaned).values()), dtype=torch.float32
        )
        one_hot = card_to_one_hot(cleaned)
        row = torch.cat([stat_tensor, one_hot, emb], dim=-1)
        feature_rows.append(row)
        cleaned_cards.append(cleaned)

    feature_matrix = torch.stack(feature_rows, dim=0) if feature_rows else torch.empty(0)
    return feature_matrix, cleaned_cards
