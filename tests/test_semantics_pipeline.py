from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.kg.io import read_jsonl
from app.semantics.builder import build_semantics
from app.semantics.enrichment import run_enrichment
from app.semantics.rule_extractors import extract_actions
from app.semantics.text import clean_card_text


class SemanticsPipelineTest(unittest.TestCase):
    def test_extract_actions_from_clean_text(self) -> None:
        text = clean_card_text("<b>Battlecry:</b> Deal 3 damage to a minion. Draw a card.")

        actions = extract_actions(text)

        self.assertEqual([action["type"] for action in actions], ["deal_damage", "draw"])
        self.assertEqual(actions[0]["amount"], 3)
        self.assertEqual(actions[0]["target"], "minion")
        self.assertEqual(actions[0]["trigger"], "battlecry")

    def test_build_semantics_includes_child_graph_and_caption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards = root / "cards.jsonl"
            metadata = root / "metadata.json"
            art_metadata = root / "art_metadata.jsonl"
            out_dir = root / "semantics"
            self._write_jsonl(
                cards,
                [
                    {
                        "id": 1,
                        "name": "Imp Caller",
                        "collectible": 1,
                        "classId": 9,
                        "cardTypeId": 4,
                        "cardSetId": 1,
                        "rarityId": 3,
                        "manaCost": 3,
                        "attack": 2,
                        "health": 2,
                        "text": "<b>Battlecry:</b> Summon two 1/1 Imps.",
                        "childIds": [2],
                        "keywordIds": [],
                        "slug": "1-imp-caller",
                    },
                    {
                        "id": 2,
                        "name": "Imp",
                        "collectible": 0,
                        "parentId": 1,
                        "classId": 9,
                        "cardTypeId": 4,
                        "cardSetId": 1,
                        "rarityId": None,
                        "manaCost": 1,
                        "attack": 1,
                        "health": 1,
                        "text": "",
                        "keywordIds": [],
                        "slug": "2-imp",
                    },
                    {
                        "id": 3,
                        "name": "Battlegrounds Spell",
                        "collectible": 0,
                        "classId": 12,
                        "cardTypeId": 42,
                        "cardSetId": 1453,
                        "manaCost": 1,
                        "text": "Discover a minion.",
                        "keywordIds": [],
                        "slug": "3-bg-spell",
                    },
                ],
            )
            metadata.write_text(json.dumps(self._metadata()), encoding="utf-8")
            self._write_jsonl(
                art_metadata,
                [
                    {
                        "file_name": "images/TEST_001.jpg",
                        "dbf_id": 1,
                        "card_id": "TEST_001",
                        "name": "Different Name Does Not Matter",
                    }
                ],
            )

            stats = build_semantics(
                cards_path=cards,
                metadata_path=metadata,
                art_metadata_path=art_metadata,
                out_dir=out_dir,
            )
            records = read_jsonl(out_dir / "cards_semantics_base.jsonl")
            captions = read_jsonl(out_dir / "lora_captions.jsonl")

        root_record = next(row for row in records if row["card_id"] == 1)
        child_record = next(row for row in records if row["card_id"] == 2)
        self.assertEqual(stats["source_cards"], 3)
        self.assertEqual(stats["excluded_special_mode_cards"], 1)
        self.assertEqual(stats["cards"], 2)
        self.assertNotIn(3, {row["card_id"] for row in records})
        self.assertEqual(root_record["child_card_ids"], [2])
        self.assertEqual(child_record["parent_card_ids"], [1])
        self.assertEqual(child_record["root_collectible_ids"], [1])
        self.assertIn("Warlock", root_record["identity"]["card_class"])
        self.assertIn("summon", [action["type"] for action in root_record["actions"]])
        self.assertIn("Hearthstone card art", root_record["lora_caption"])
        self.assertEqual(root_record["source"]["art_image"], "images/TEST_001.jpg")
        self.assertEqual([row["image"] for row in captions], ["images/TEST_001.jpg"])

    def test_build_semantics_can_include_special_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards = root / "cards.jsonl"
            out_dir = root / "semantics"
            self._write_jsonl(
                cards,
                [
                    {
                        "id": 3,
                        "name": "Battlegrounds Spell",
                        "collectible": 0,
                        "classId": 12,
                        "cardTypeId": 42,
                        "cardSetId": 1453,
                        "manaCost": 1,
                        "text": "Discover a minion.",
                        "keywordIds": [],
                        "slug": "3-bg-spell",
                    }
                ],
            )

            stats = build_semantics(
                cards_path=cards,
                metadata_path=None,
                art_metadata_path=None,
                exclude_special_modes=False,
                out_dir=out_dir,
            )
            records = read_jsonl(out_dir / "cards_semantics_base.jsonl")

        self.assertEqual(stats["excluded_special_mode_cards"], 0)
        self.assertEqual([row["card_id"] for row in records], [3])

    def test_enrichment_dry_run_writes_merged_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantics = root / "cards_semantics_base.jsonl"
            out_dir = root / "enriched"
            prompt = root / "prompt.md"
            self._write_jsonl(
                semantics,
                [
                    {
                        "card_id": 1,
                        "name": "Card",
                        "identity": {"card_class": ["Mage"], "card_type": "Spell"},
                        "stats": {},
                        "text": {"clean": "Deal 1 damage."},
                        "keywords": [],
                        "actions": [{"type": "deal_damage", "amount": 1, "target": "target"}],
                        "mechanic_tags": ["deal_damage"],
                        "visual_tags": ["arcane magic"],
                        "child_card_ids": [],
                        "derived_cards": [],
                        "slug": "1-card",
                    }
                ],
            )
            prompt.write_text("Cards:\n{{CARDS_JSON}}", encoding="utf-8")

            stats = run_enrichment(
                semantics_path=semantics,
                out_dir=out_dir,
                prompt_template=prompt,
                dry_run=True,
                resume=False,
                force_llm=True,
            )
            records = read_jsonl(out_dir / "cards_semantics_enriched.jsonl")

        self.assertEqual(stats["cards"], 1)
        self.assertEqual(records[0]["enrichment"]["status"], "base_only")
        self.assertIn("Mage", records[0]["lora_caption"])

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _metadata(self) -> dict:
        return {
            "maps": {
                "cardTypeId": {"4": {"id": 4, "name": "Minion"}},
                "cardSetId": {"1": {"id": 1, "name": "Core"}},
                "classId": {"9": {"id": 9, "name": "Warlock"}},
                "rarityId": {"3": {"id": 3, "name": "Rare"}},
                "spellSchoolId": {},
                "minionTypeId": {},
                "keywordIds": {},
            }
        }


if __name__ == "__main__":
    unittest.main()
