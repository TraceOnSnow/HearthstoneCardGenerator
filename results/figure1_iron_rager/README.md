# Figure 1 Case: Iron Rager

## User Request
I want to DIY a new Rager meme card inspired by Magma Rager's classic 3-mana 5/1 statline, but make it feel like a defensive Control Warrior card that gains Armor. Keep it simple and plausible, like something Hearthstone players would actually design.

## Structured Query
```json
{
  "query_id": "diy_rager_meme_en",
  "text": "I want to DIY a new Rager meme card inspired by Magma Rager's classic 3-mana 5/1 statline, but make it feel like a defensive Control Warrior card that gains Armor. Keep it simple and plausible, like something Hearthstone players would actually design.",
  "classes": [
    "Warrior"
  ],
  "card_types": [
    "Minion"
  ],
  "keywords": [
    "Rager",
    "Armor"
  ],
  "actions": [
    "gain_armor"
  ],
  "targets": [],
  "resources": [
    "3"
  ],
  "spell_schools": [],
  "minion_types": [],
  "mechanic_tags": [
    "Rager"
  ],
  "constraints": [
    "5/1 statline inspired by Magma Rager",
    "defensive Control Warrior theme",
    "armor gain",
    "simple and plausible meme design"
  ],
  "generated_roles": [
    "Control Warrior defensive minion",
    "Rager meme card"
  ],
  "generated_card_names": [],
  "related_card_names": [
    "Magma Rager",
    "Rager"
  ],
  "triggers": [],
  "conditions": [],
  "generation_hints": {
    "visual_tags": [
      "warrior aesthetic",
      "armor theme",
      "elemental reminiscent",
      "simple meme-style design"
    ]
  }
}
```

## Selected KG Evidence
- Magma Rager (CS2_118, dbf_id=1653): family/statline anchor: classic 3-mana 5/1 Rager meme
- Hookfist-3000 (NX2_028, dbf_id=86692): mechanic anchor: Warrior minion that gains Armor
- Armorsmith (EX1_402, dbf_id=596): class identity: iconic Warrior Armor synergy
- Eternium Rover (BOT_059, dbf_id=48691): defensive Warrior flavor: armored mech/minion reference

## Designed Card
**Iron Rager** - 3 mana Warrior Minion, 5/1
Text: Battlecry: Gain 3 Armor.
Flavor: Molten fury tempered by iron will.

## LoRA Caption
Hearthstone card art, a burly elemental rager forged from molten rock and clad in iron armor, Warrior class, glowing embers and metallic plates.

## Figure Replacement Notes
- Use `figure1_iron_rager_panel.png` as a ready-made Figure 1 draft if time is short.
- Replace generation placeholder boxes with real SD/LoRA outputs if available.
- Keep the Magma Rager + Hookfist evidence contrast because it explains why KG/faceted retrieval matters.