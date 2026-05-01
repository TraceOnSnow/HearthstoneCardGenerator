from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import make_judging_template, run_baseline_retrieval, summarize_judging


class RetrievalEvalScriptsTest(unittest.TestCase):
    def test_tfidf_baseline_prefers_matching_caption(self) -> None:
        queries = [
            {
                "query_id": "mage_freeze_spell",
                "text": "Mage spell that freezes enemies",
                "classes": ["Mage"],
                "card_types": ["Spell"],
                "keywords": ["Freeze"],
                "actions": ["freeze"],
                "visual_tags": ["frost magic"],
            }
        ]
        captions = [
            {
                "card_id": 1,
                "name": "Frost Bolt",
                "image": "images/frost.jpg",
                "caption": "Hearthstone card art, Mage, Frost spell, Freeze enemy",
            },
            {
                "card_id": 2,
                "name": "Armor Up",
                "image": "images/armor.jpg",
                "caption": "Hearthstone card art, Warrior, gain armor",
            },
        ]

        results = run_baseline_retrieval.retrieve(queries, captions, top_k=1)

        self.assertEqual(results[0]["card_id"], 1)
        self.assertEqual(results[0]["method"], "tfidf_baseline")

    def test_judging_template_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_path = root / "baseline.jsonl"
            template_path = root / "judging.csv"
            summary_path = root / "summary.csv"
            results_path.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "method": "tfidf_baseline",
                        "rank": 1,
                        "card_id": 10,
                        "card_name": "Card",
                        "image": "images/card.jpg",
                        "query_text": "query",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = [make_judging_template.template_row(row) for row in make_judging_template.read_jsonl(results_path)]
            with template_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=make_judging_template.FIELDNAMES)
                writer.writeheader()
                rows[0]["class_match"] = "1"
                rows[0]["action_match"] = "0"
                rows[0]["keyword_match"] = "1"
                rows[0]["overall_relevance"] = "2"
                writer.writerows(rows)

            summarize_judging.main_with_paths(template_path, summary_path)

            with summary_path.open("r", encoding="utf-8", newline="") as f:
                summary = list(csv.DictReader(f))

        self.assertEqual(summary[0]["method"], "tfidf_baseline")
        self.assertEqual(summary[0]["class_match_at_5"], "1.0000")
        self.assertEqual(summary[0]["overall_relevance_at_5"], "2.0000")


if __name__ == "__main__":
    unittest.main()
