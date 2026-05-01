from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.diffusion.lora_data import build_caption, normalize_lora_rows


class LoraDataTest(unittest.TestCase):
    def test_normalize_lora_rows_resolves_images_and_adds_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "dataset"
            image_path = image_root / "images" / "card.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"not-a-real-image")
            metadata = root / "captions.jsonl"
            self._write_jsonl(
                metadata,
                [
                    {
                        "image": "images/card.jpg",
                        "caption": "Hearthstone card art, Mage, Spell",
                    }
                ],
            )

            rows, missing = normalize_lora_rows(metadata_path=metadata, image_root=image_root)

        self.assertEqual(missing, [])
        self.assertEqual(rows[0]["image_path"], str(image_path.resolve()))
        self.assertEqual(rows[0]["caption"], "hsart Hearthstone card art, Mage, Spell")

    def test_build_caption_falls_back_to_metadata_shape(self) -> None:
        caption = build_caption(
            {
                "name": "Arcane Bolt",
                "card_class": "Mage",
                "type": "Spell",
                "set": "Core",
                "artist": "Artist",
            },
            caption_column="caption",
            trigger_token="hsart",
        )

        self.assertEqual(caption, "hsart Hearthstone card art, Arcane Bolt, Mage, Spell, Core, Artist")

    def test_normalize_lora_rows_accepts_hf_file_name_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "images" / "card.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"not-a-real-image")
            metadata = root / "metadata.jsonl"
            self._write_jsonl(
                metadata,
                [
                    {
                        "file_name": "images/card.jpg",
                        "text": "hsart Hearthstone card art, Mage spell",
                    }
                ],
            )

            rows, missing = normalize_lora_rows(metadata_path=metadata, image_root=root)

        self.assertEqual(missing, [])
        self.assertEqual(rows[0]["image_path"], str(image_path.resolve()))
        self.assertEqual(rows[0]["caption"], "hsart Hearthstone card art, Mage spell")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    unittest.main()
