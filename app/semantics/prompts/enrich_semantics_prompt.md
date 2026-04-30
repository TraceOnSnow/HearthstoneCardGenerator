You enrich structured Hearthstone card semantics for image-generation captions and KG retrieval.

Return strict JSON only:

{
  "cards": [
    {
      "card_id": 123,
      "actions": [
        {
          "type": "deal_damage",
          "amount": 3,
          "target": "enemy_minion",
          "target_scope": "enemy",
          "resource": "health",
          "condition": "holding_a_dragon",
          "trigger": "battlecry",
          "duration": null,
          "raw_phrase": "Battlecry: If you're holding a Dragon, deal 3 damage."
        }
      ],
      "mechanic_tags": ["battlecry_damage", "dragon_synergy"],
      "visual_tags": ["arcane projectile", "dragon-themed magic"],
      "derived_cards": [
        {"card_id": 456, "relation": "HAS_CHILD_CARD", "role": "summoned_token"}
      ],
      "semantic_summary": "A concise one-sentence semantic description focused on mechanics and visual meaning."
    }
  ]
}

Rules:
- Do not invent card IDs, names, stats, classes, or child card IDs.
- Keep action type names controlled and snake_case.
- Prefer visual tags that help Hearthstone-style artwork generation.
- Keep visual tags concrete, short, and non-photographic.
- Preserve base actions unless they are clearly wrong; add missing actions for complex text.
- If a field cannot be improved, return the base value.

Cards:
{{CARDS_JSON}}

