You are an information extraction engine.

Task:
Convert the given Hearthstone card list into a strict knowledge graph entity structure.
Only use information that is explicit in card fields.
Do not hallucinate missing facts.

Input cards include these fields only:
- id
- name
- text
- manaCost
- attack
- health

Output requirements:
1) Return valid JSON only.
2) Top-level schema must be:
{
  "cards": [
    {
      "card_id": <int>,
      "name": <string>,
      "attributes": {
        "manaCost": <int or null>,
        "attack": <int or null>,
        "health": <int or null>
      },
      "mechanics": [<string>],
      "entities": [
        {
          "type": <"keyword"|"effect"|"target"|"condition"|"action"|"other">,
          "name": <string>
        }
      ],
      "relations": [
        {
          "subject": <string>,
          "predicate": <string>,
          "object": <string>
        }
      ]
    }
  ]
}

3) Subject/object naming convention:
- Card node: "card:<card_id>"
- Mechanic node: "mechanic:<normalized_name>"
- Generic entity node: "entity:<type>:<normalized_name>"

4) Relation conventions:
- card HAS_MECHANIC mechanic
- card HAS_ENTITY entity

Cards:
{{CARDS_JSON}}
