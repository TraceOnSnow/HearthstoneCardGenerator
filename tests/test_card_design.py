from __future__ import annotations

import unittest

from app.card_design.kg_designer import build_evidence_package


class CardDesignEvidenceTest(unittest.TestCase):
    def test_evidence_package_keeps_low_rank_family_anchor(self) -> None:
        query = {
            "related_card_names": ["Magma Rager", "Rager"],
        }
        evidence = [
            self._row(1, "Hookfist-3000", ["classes=Warrior", "card_types=Minion", "actions=gain_armor"]),
            self._row(2, "Armorsmith", ["classes=Warrior", "card_types=Minion", "actions=gain_armor"]),
            self._row(20, "Magma Rager", ["related_name_match"]),
        ]

        package = build_evidence_package(query=query, evidence=evidence)

        family_names = [row["name"] for row in package["facets"]["family_or_named_anchors"]]
        mechanic_names = [row["name"] for row in package["facets"]["mechanic_matches"]]
        self.assertIn("Magma Rager", family_names)
        self.assertIn("Hookfist-3000", mechanic_names)

    def test_diversified_shortlist_deduplicates_same_reason_signature(self) -> None:
        query = {"related_card_names": ["Rager"]}
        evidence = [
            self._row(1, "Armor Card A", ["classes=Warrior", "card_types=Minion", "actions=gain_armor", "text_overlap=4"]),
            self._row(2, "Armor Card B", ["classes=Warrior", "card_types=Minion", "actions=gain_armor", "text_overlap=3"]),
            self._row(3, "Magma Rager", ["related_name_match"]),
        ]

        package = build_evidence_package(query=query, evidence=evidence)
        names = [row["name"] for row in package["facets"]["diversified_shortlist"]]

        self.assertIn("Magma Rager", names)
        self.assertEqual(1, len([name for name in names if name.startswith("Armor Card")]))

    def test_evidence_package_records_lora_reference_policy(self) -> None:
        package = build_evidence_package(query={}, evidence=[])

        note = package["retrieval_policy"]["lora_reference_note"]
        self.assertIn("one primary reference", note)
        self.assertIn("0.75-0.85", note)

    def _row(self, rank: int, name: str, reasons: list[str]) -> dict:
        return {
            "rank": rank,
            "score": 10.0 / rank,
            "card_id": rank,
            "name": name,
            "reasons": reasons,
        }


if __name__ == "__main__":
    unittest.main()
