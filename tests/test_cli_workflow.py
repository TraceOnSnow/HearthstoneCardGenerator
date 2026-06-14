from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.workflow import GenerateOptions, run_generate


class CliWorkflowTest(unittest.TestCase):
    def test_mock_generate_writes_standard_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            card_index = root / "card_index.jsonl"
            semantics = root / "semantics.jsonl"
            out_dir = root / "run"
            card_index.write_text(
                json.dumps(
                    {
                        "card_id": 1,
                        "name": "Arcane Golem",
                        "image": "images/arcane_golem.jpg",
                        "text": "Battlecry: Gain Mana Crystals.",
                        "node_ids": ["class:mage", "card_type:minion", "minion_type:mech", "action:discover"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            semantics.write_text(
                json.dumps(
                    {
                        "card_id": 1,
                        "identity": {"class": ["Mage"], "card_type": "Minion"},
                        "stats": {"mana_cost": 3, "attack": 4, "health": 4},
                        "text": {"clean": "Battlecry: Gain Mana Crystals."},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_generate(
                GenerateOptions(
                    request_text="Create a Mage Mech minion with Battlecry and Discover.",
                    out_dir=out_dir,
                    card_index_path=card_index,
                    semantics_path=semantics,
                    mock_design=True,
                    image_provider="mock",
                )
            )

            self.assertEqual(out_dir, result["out_dir"])
            for name in [
                "input.json",
                "query.json",
                "retrieved_cards.json",
                "card.json",
                "design.json",
                "art_prompt.txt",
                "art.png",
                "final_card.png",
                "run.json",
                "summary.md",
            ]:
                self.assertTrue((out_dir / name).exists(), name)
            card = json.loads((out_dir / "card.json").read_text(encoding="utf-8"))
            self.assertEqual("Mage", card["class"][0])
            self.assertEqual("Minion", card["card_type"])


if __name__ == "__main__":
    unittest.main()
