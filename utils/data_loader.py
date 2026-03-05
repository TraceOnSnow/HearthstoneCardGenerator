from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor


class HearthstoneCardDataset(Dataset):
    """PyTorch Dataset for Hearthstone card images and metadata.

    Each sample contains:
        - ``image``: Preprocessed card image tensor.
        - ``input_ids`` / ``attention_mask``: Tokenised card description.
        - ``label``: Integer card-type label (optional).
        - ``metadata``: Raw JSON metadata dict.
    """

    def __init__(
        self,
        data_dir: str | Path,
        processor: CLIPProcessor,
        split: str = "train",
        image_subdir: str = "images",
        metadata_file: str = "cards.json",
    ) -> None:
        """Initialise the dataset.

        Args:
            data_dir: Root directory containing the dataset.
            processor: CLIP processor used to encode images and text.
            split: One of ``"train"``, ``"val"``, or ``"test"``.
            image_subdir: Sub-directory (inside *data_dir*) holding images.
            metadata_file: JSON file with card metadata relative to *data_dir*.
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / image_subdir
        self.processor = processor
        self.split = split

        metadata_path = self.data_dir / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as fh:
            all_cards: list[dict[str, Any]] = json.load(fh)

        self.cards = self._filter_split(all_cards, split)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_split(cards: list[dict], split: str) -> list[dict]:
        """Return cards belonging to the requested split.

        Expects each card dict to have a ``"split"`` key.  If the key is
        absent all cards are returned for the ``"train"`` split and an
        empty list for others.
        """
        if not cards:
            return []
        if "split" not in cards[0]:
            return cards if split == "train" else []
        return [c for c in cards if c.get("split") == split]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cards)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        card = self.cards[idx]

        # Load image
        image_path = self.image_dir / card.get("image", f"{card['id']}.png")
        image = None
        if image_path.exists():
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                card_id = card.get("id", idx)
                raise OSError(
                    f"Failed to load image for card '{card_id}' at '{image_path}': {exc}"
                ) from exc

        # Build description text
        description = card.get("text", card.get("name", ""))

        # Process via CLIP
        encoding = self.processor(
            images=image,
            text=description,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        sample: dict[str, Any] = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "metadata": card,
        }
        if image is not None:
            sample["pixel_values"] = encoding["pixel_values"].squeeze(0)
        if "label" in card:
            sample["label"] = torch.tensor(card["label"], dtype=torch.long)

        return sample


class HearthstoneDataLoader:
    """Factory for creating PyTorch DataLoaders for Hearthstone card data."""

    def __init__(
        self,
        data_dir: str | Path,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        """Initialise the data loader factory.

        Args:
            data_dir: Root directory containing the dataset.
            clip_model_name: HuggingFace CLIP model identifier used to
                instantiate the processor.
            batch_size: Samples per mini-batch.
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory (recommended for GPU training).
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

    def get_loader(self, split: str = "train", shuffle: bool | None = None) -> DataLoader:
        """Return a DataLoader for the requested split.

        Args:
            split: One of ``"train"``, ``"val"``, or ``"test"``.
            shuffle: Whether to shuffle data.  Defaults to ``True`` for
                ``"train"`` and ``False`` otherwise.

        Returns:
            Configured :class:`torch.utils.data.DataLoader`.
        """
        if shuffle is None:
            shuffle = split == "train"

        dataset = HearthstoneCardDataset(
            data_dir=self.data_dir,
            processor=self.processor,
            split=split,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_all_loaders(self) -> dict[str, DataLoader]:
        """Return DataLoaders for all three splits.

        Returns:
            Dictionary with keys ``"train"``, ``"val"``, ``"test"``.
        """
        return {
            "train": self.get_loader("train"),
            "val": self.get_loader("val", shuffle=False),
            "test": self.get_loader("test", shuffle=False),
        }
