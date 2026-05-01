You enrich structured Hearthstone card semantics for KG retrieval and LoRA captions.

Return strict JSON only. Do not use markdown. Do not explain the schema. Keep reasoning brief internally and put only the final JSON in the answer.

Task:
- Preserve deterministic facts unless clearly wrong.
- Add gameplay semantics that rules miss: generated cards, summoned tokens, choice options, follow-up cards, conditions, durations, triggers, constraints, and implicit mechanic tags.
- Use child_cards from the input to label derived_cards roles. Never invent card IDs.
- Keep mechanic_tags gameplay-only. Visual hints must go under generation_hints.visual_tags and are not KG facts.

Output schema:
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
      "action_groups": [
        {
          "type": "choose_one",
          "raw_phrase": "Choose One - ...",
          "action_indices": [0, 1],
          "options": ["option_a", "option_b"]
        }
      ],
      "mechanic_tags": ["battlecry_damage", "dragon_synergy"],
      "constraints": ["requires_holding_dragon"],
      "generated_card_refs": [
        {"card_id": 456, "name": "Solar Eclipse", "role": "generated_spell", "evidence": "text evidence"}
      ],
      "derived_cards": [
        {"card_id": 789, "relation": "HAS_CHILD_CARD", "role": "summoned_token", "evidence": "text evidence"}
      ],
      "related_card_refs": [
        {"card_id": null, "name": "Named Card", "relation": "MENTIONS_OR_GENERATES", "evidence": "text evidence"}
      ],
      "semantic_summary": "One concise sentence about gameplay semantics.",
      "generation_hints": {"visual_tags": ["optional LoRA-only visual hint"]}
    }
  ]
}

Controlled action types:
deal_damage, heal, gain_armor, summon, draw, discover, add_to_hand, destroy, transform, copy, shuffle_into_deck, discard, resurrect, freeze, silence, equip, attack, give_buff, gain_attack, gain_health, set_stats, reduce_cost, increase_cost, gain_mana_crystal, refresh_mana, overload, spend_corpse, generate_corpse, cast_spell, trigger_deathrattle, return_to_hand, swap, steal, excavate, forge, trade, dredge, recruit, other.

Common constraints:
cannot_attack, next_turn_only, this_turn_only, random_target, enemy_only, friendly_only, while_in_hand, requires_spell_school, requires_card_type, requires_tribe.

Cards:
{{CARDS_JSON}}
