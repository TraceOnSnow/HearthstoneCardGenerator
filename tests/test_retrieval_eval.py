from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from app.retrieval.evaluation import summarize_judging, write_judging_template
from app.retrieval.tfidf import retrieve_tfidf


class RetrievalEvalTest(unittest.TestCase):
    def test_tfidf_retrieves_matching_caption(self) -> None:
        corpus = [
            {"card_id": 1, "card_name": "Armor Card", "image": "images/a.jpg", "caption": "Warrior gain armor defense"},
            {"card_id": 2, "card_name": "Fire Card", "image": "images/b.jpg", "caption": "Mage fire spell damage"},
        ]
        queries = [{"query_id": "armor", "text": "Warrior card that gains Armor", "classes": ["Warrior"], "actions": ["gain_armor"]}]

        rows = retrieve_tfidf(corpus=corpus, queries=queries, top_k=1)

        self.assertEqual(rows[0]["card_id"], 1)
        self.assertEqual(rows[0]["method"], "tfidf_baseline")
        self.assertEqual(rows[0]["rank"], 1)

    def test_judging_template_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "judging.csv"
            summary = root / "summary.csv"
            write_judging_template(
                [
                    {
                        "query_id": "q",
                        "method": "semantic_kg",
                        "rank": 1,
                        "card_id": 1,
                        "card_name": "Card",
                        "query_text": "query",
                        "image": "images/a.jpg",
                        "score": 1.0,
                    }
                ],
                template,
            )
            with template.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            rows[0]["class_match"] = "1"
            rows[0]["action_match"] = "1"
            rows[0]["keyword_match"] = "0"
            rows[0]["overall_relevance"] = "2"
            with template.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            result = summarize_judging(template, summary)

        self.assertEqual(result[0]["method"], "semantic_kg")
        self.assertEqual(result[0]["overall_relevance_at_5"], 2.0)
        self.assertEqual(result[0]["hit_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
