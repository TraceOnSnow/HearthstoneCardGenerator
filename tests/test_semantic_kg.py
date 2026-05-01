from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.kg.io import read_jsonl
from app.semantic_kg.build import build_semantic_kg, graph_from_semantics
from app.semantic_kg.query_parser import parse_query_rule
from app.semantic_kg.retrieval import retrieve_one


class SemanticKgTest(unittest.TestCase):
    def test_graph_from_semantics_builds_nodes_edges_and_card_index(self) -> None:
        graph = graph_from_semantics([self._health_drink()])

        node_ids = {node["id"] for node in graph["nodes"]}
        edge_keys = {(edge["source"], edge["predicate"], edge["target"]) for edge in graph["edges"]}
        card_index = graph["card_index"][0]

        self.assertIn("card:107923", node_ids)
        self.assertIn("class:warlock", node_ids)
        self.assertIn("action:deal_damage", node_ids)
        self.assertNotIn("visual:fel_magic", node_ids)
        self.assertIn(("card:107923", "HAS_CLASS", "class:warlock"), edge_keys)
        self.assertIn(("card:107923", "PERFORMS_ACTION", "action:deal_damage"), edge_keys)
        self.assertIn("action:deal_damage", card_index["node_ids"])
        self.assertEqual(card_index["image"], "images/VAC_951.jpg")
        self.assertNotIn("caption", card_index)

    def test_retrieve_one_scores_structured_query(self) -> None:
        graph = graph_from_semantics([self._health_drink(), self._mage_spell()])
        query = {
            "query_id": "warlock_lifesteal_damage",
            "text": "Warlock spell that deals damage and has Lifesteal",
            "classes": ["Warlock"],
            "card_types": ["Spell"],
            "keywords": ["Lifesteal"],
            "actions": ["deal_damage"],
            "targets": ["minion"],
            "spell_schools": ["Fel"],
            "mechanic_tags": ["lifesteal_damage"],
        }

        results = retrieve_one(graph["card_index"], query=query, top_k=1)

        self.assertEqual(results[0]["card_id"], 107923)
        self.assertEqual(results[0]["rank"], 1)
        self.assertIn("classes=Warlock", results[0]["reasons"])
        self.assertIn("actions=deal_damage", results[0]["reasons"])

    def test_parse_query_rule_extracts_common_fields(self) -> None:
        query = parse_query_rule("I want a dark Warlock spell that drains life from a minion.")

        self.assertIn("Warlock", query["classes"])
        self.assertIn("Spell", query["card_types"])
        self.assertNotIn("Mage", query["classes"])
        self.assertNotIn("Minion", query["card_types"])
        self.assertIn("deal_damage", query["actions"])
        self.assertIn("minion", query["targets"])
        self.assertIn("dark magic", query["generation_hints"]["visual_tags"])

    def test_graph_links_generated_card_refs_and_constraints(self) -> None:
        root = self._health_drink()
        root["card_id"] = 61605
        root["name"] = "Kiri, Chosen of Elune"
        root["constraints"] = ["battlecry_only"]
        root["generated_card_refs"] = [
            {"card_id": 61450, "name": "Solar Eclipse", "role": "generated_spell", "evidence": "Add a Solar Eclipse to your hand."}
        ]
        child = self._mage_spell()
        child["card_id"] = 61450
        child["name"] = "Solar Eclipse"

        graph = graph_from_semantics([root, child])
        edge_keys = {(edge["source"], edge["predicate"], edge["target"]) for edge in graph["edges"]}
        node_ids = {node["id"] for node in graph["nodes"]}

        self.assertIn("constraint:battlecry_only", node_ids)
        self.assertIn("generated_role:generated_spell", node_ids)
        self.assertIn(("card:61605", "GENERATES_CARD", "card:61450"), edge_keys)

    def test_build_semantic_kg_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantics = root / "semantics.jsonl"
            out_dir = root / "kg"
            self._write_jsonl(semantics, [self._health_drink()])

            stats = build_semantic_kg(semantics_path=semantics, out_dir=out_dir)
            nodes = read_jsonl(out_dir / "nodes.jsonl")
            edges = read_jsonl(out_dir / "edges.jsonl")
            cards = read_jsonl(out_dir / "card_index.jsonl")

        self.assertEqual(stats["cards"], 1)
        self.assertTrue(nodes)
        self.assertTrue(edges)
        self.assertEqual(cards[0]["card_id"], 107923)

    def _health_drink(self) -> dict:
        return {
            "card_id": 107923,
            "slug": "107923-health-drink",
            "name": '"Health" Drink',
            "collectible": True,
            "is_derived": False,
            "root_collectible_ids": [107923],
            "parent_card_ids": [],
            "child_card_ids": [],
            "derivation_depth": 0,
            "identity": {
                "card_type": "Spell",
                "card_class": ["Warlock"],
                "set": "Perils in Paradise",
                "rarity": "Rare",
                "spell_school": "Fel",
                "minion_type": None,
                "artist": "Vladimir Kafanov",
            },
            "stats": {"mana_cost": 3, "attack": None, "health": None},
            "text": {"clean": "Lifesteal. Deal 3 damage to a minion."},
            "keywords": ["Lifesteal"],
            "actions": [
                {
                    "type": "deal_damage",
                    "amount": 3,
                    "target": "minion",
                    "resource": "health",
                    "trigger": "on_play",
                    "raw_phrase": "Deal 3 damage to a minion.",
                }
            ],
            "mechanic_tags": ["deal_damage", "lifesteal_damage"],
            "visual_tags": ["hearthstone fantasy art", "warlock", "fel magic"],
            "source": {"art_image": "images/VAC_951.jpg"},
            "lora_caption": "Hearthstone card art, Warlock, Fel spell, Lifesteal",
        }

    def _mage_spell(self) -> dict:
        row = self._health_drink()
        row = json.loads(json.dumps(row))
        row["card_id"] = 2539
        row["name"] = "Flame Lance"
        row["identity"]["card_class"] = ["Mage"]
        row["identity"]["spell_school"] = "Fire"
        row["keywords"] = []
        row["mechanic_tags"] = ["deal_damage"]
        row["visual_tags"] = ["hearthstone fantasy art", "fire magic"]
        row["source"]["art_image"] = "images/AT_001.jpg"
        return row

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    unittest.main()
