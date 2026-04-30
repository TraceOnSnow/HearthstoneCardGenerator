from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.kg.graph import build_graph, parse_json_response
from app.kg.io import load_cards, select_cards
from app.kg.models import KgRunConfig
from app.kg.pipeline import run_pipeline
from app.kg.prompting import build_prompt_rows


class KgPipelineTest(unittest.TestCase):
    def test_load_cards_filters_collectible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cards.jsonl"
            self._write_jsonl(
                path,
                [
                    {"id": 1, "name": "Keep", "collectible": 1},
                    {"id": 2, "name": "Skip", "collectible": 0},
                    {"name": "Missing id", "collectible": 1},
                ],
            )

            cards = load_cards(path)

        self.assertEqual([card.id for card in cards], [1])

    def test_prompt_rows_are_chunked(self) -> None:
        cards = self._cards(3)

        rows = build_prompt_rows(cards, template="Cards:\n{{CARDS_JSON}}", chunk_size=2)

        self.assertEqual([row["card_count"] for row in rows], [2, 1])
        self.assertEqual(rows[0]["card_ids"], [1, 2])
        self.assertIn('"name": "Card 1"', rows[0]["prompt"])

    def test_build_graph_includes_explicit_and_llm_edges(self) -> None:
        cards = self._cards(1)
        cards[0].keywordIds.append(38)
        llm_outputs = [
            {
                "batch_id": 1,
                "status": "ok",
                "raw_response": json.dumps(
                    {
                        "cards": [
                            {
                                "card_id": 1,
                                "name": "Card 1",
                                "attributes": {},
                                "mechanics": ["Lifesteal"],
                                "entities": [{"type": "action", "name": "Draw"}],
                                "relations": [],
                            }
                        ]
                    }
                ),
            }
        ]

        graph = build_graph(cards, llm_outputs, metadata=self._metadata())
        node_ids = {node["id"] for node in graph["nodes"]}
        node_names = {node["id"]: node["name"] for node in graph["nodes"]}
        edge_keys = {(edge["source"], edge["predicate"], edge["target"]) for edge in graph["edges"]}

        self.assertIn("card:1", node_ids)
        self.assertIn("keyword:38", node_ids)
        self.assertIn("mechanic:lifesteal", node_ids)
        self.assertEqual(node_names["card_type:4"], "Minion")
        self.assertEqual(node_names["keyword:38"], "Lifesteal")
        self.assertIn(("card:1", "HAS_KEYWORD", "keyword:38"), edge_keys)
        self.assertIn(("card:1", "HAS_MECHANIC", "mechanic:lifesteal"), edge_keys)

    def test_build_graph_includes_semantic_llm_edges(self) -> None:
        cards = self._cards(1)
        llm_outputs = [
            {
                "batch_id": 1,
                "status": "ok",
                "raw_response": json.dumps(
                    {
                        "cards": [
                            {
                                "card_id": 1,
                                "name": "Card 1",
                                "explicit_keywords": ["Battlecry"],
                                "actions": [
                                    {
                                        "type": "deal_damage",
                                        "amount": 3,
                                        "target": "enemy_minion",
                                        "condition": "battlecry",
                                        "resource": "health",
                                        "raw_phrase": "Deal 3 damage to an enemy minion.",
                                    }
                                ],
                                "resources": [
                                    {"type": "health", "operation": "affect", "amount": 3}
                                ],
                                "tribal_or_school_references": ["Fire"],
                                "synergy_tags": ["damage_synergy"],
                                "constraints": ["battlecry_only"],
                                "raw_phrases": ["Deal 3 damage"],
                            }
                        ]
                    }
                ),
            }
        ]

        graph = build_graph(cards, llm_outputs, metadata=self._metadata())
        node_ids = {node["id"] for node in graph["nodes"]}
        edge_keys = {(edge["source"], edge["predicate"], edge["target"]) for edge in graph["edges"]}
        action_edges = [
            edge
            for edge in graph["edges"]
            if edge["source"] == "card:1" and edge["predicate"] == "PERFORMS_ACTION"
        ]

        self.assertIn("keyword_text:battlecry", node_ids)
        self.assertIn("action:deal_damage", node_ids)
        self.assertIn("target:enemy_minion", node_ids)
        self.assertIn("resource:health", node_ids)
        self.assertIn("condition:battlecry", node_ids)
        self.assertIn("synergy:damage_synergy", node_ids)
        self.assertIn(("card:1", "TARGETS", "target:enemy_minion"), edge_keys)
        self.assertEqual(action_edges[0]["attributes"]["amount"], 3)
        self.assertEqual(action_edges[0]["attributes"]["target_label"], "enemy_minion")

    def test_parse_json_response_accepts_code_fence(self) -> None:
        parsed = parse_json_response('```json\n{"cards": []}\n```')
        self.assertEqual(parsed, {"cards": []})

    def test_parse_json_response_accepts_wrapped_json(self) -> None:
        parsed = parse_json_response('Here is the JSON:\n{"cards": []}\nDone.')
        self.assertEqual(parsed, {"cards": []})

    def test_smoke_pipeline_writes_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "cards.jsonl"
            out_dir = root / "kg"
            self._write_jsonl(source, [self._raw_card(1), self._raw_card(2)])

            stats = run_pipeline(
                KgRunConfig(
                    source_jsonl=str(source),
                    out_dir=str(out_dir),
                    metadata_json=None,
                    prompt_template="app/kg/prompts/kg_entity_extraction_prompt.md",
                    limit=1,
                    sample_size=None,
                    random_seed=42,
                    chunk_size=1,
                    dry_run=True,
                    provider="google",
                    model="gemini-2.5-flash-lite",
                    temperature=0.1,
                    timeout_seconds=60,
                    resume=False,
                    force_llm=True,
                    visualize=False,
                )
            )

            graph = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))

        self.assertEqual(stats["cards"], 1)
        self.assertEqual(graph["stats"]["cards"], 1)

    def test_select_cards_sample_is_deterministic(self) -> None:
        cards = self._cards(10)
        first = select_cards(cards, limit=None, sample_size=4, seed=7)
        second = select_cards(cards, limit=None, sample_size=4, seed=7)
        self.assertEqual([card.id for card in first], [card.id for card in second])

    def _cards(self, count: int):
        path = None
        rows = [self._raw_card(idx) for idx in range(1, count + 1)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cards.jsonl"
            self._write_jsonl(path, rows)
            return load_cards(path)

    def _raw_card(self, card_id: int) -> dict:
        return {
            "id": card_id,
            "name": f"Card {card_id}",
            "text": "<b>Battlecry:</b> Draw a card.",
            "collectible": 1,
            "manaCost": 2,
            "attack": 1,
            "health": 3,
            "cardTypeId": 4,
            "cardSetId": 1,
            "classId": 2,
            "rarityId": 3,
            "spellSchoolId": None,
            "minionTypeId": None,
            "multiClassIds": [],
            "keywordIds": [],
            "childIds": [],
            "artistName": "Artist",
            "slug": f"{card_id}-card",
        }

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _metadata(self) -> dict:
        return {
            "maps": {
                "cardTypeId": {"4": {"id": 4, "name": "Minion"}},
                "cardSetId": {"1": {"id": 1, "name": "Core"}},
                "classId": {"2": {"id": 2, "name": "Druid"}},
                "rarityId": {"3": {"id": 3, "name": "Rare"}},
                "spellSchoolId": {},
                "minionTypeId": {},
                "keywordIds": {"38": {"id": 38, "name": "Lifesteal"}},
            }
        }


if __name__ == "__main__":
    unittest.main()
