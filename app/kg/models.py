from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class CardRecord:
    id: int
    name: str
    text: str
    manaCost: int | None
    attack: int | None
    health: int | None
    collectible: int | None
    cardTypeId: int | None
    cardSetId: int | None
    classId: int | None
    rarityId: int | None
    spellSchoolId: int | None
    minionTypeId: int | None
    multiClassIds: list[int]
    keywordIds: list[int]
    childIds: list[int]
    artistName: str
    slug: str

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "CardRecord | None":
        card_id = raw.get("id")
        if not isinstance(card_id, int):
            return None

        return cls(
            id=card_id,
            name=str(raw.get("name", "")).strip(),
            text=str(raw.get("text", "")).strip(),
            manaCost=_optional_int(raw.get("manaCost")),
            attack=_optional_int(raw.get("attack")),
            health=_optional_int(raw.get("health")),
            collectible=_optional_int(raw.get("collectible")),
            cardTypeId=_optional_int(raw.get("cardTypeId")),
            cardSetId=_optional_int(raw.get("cardSetId")),
            classId=_optional_int(raw.get("classId")),
            rarityId=_optional_int(raw.get("rarityId")),
            spellSchoolId=_optional_int(raw.get("spellSchoolId")),
            minionTypeId=_optional_int(raw.get("minionTypeId")),
            multiClassIds=_int_list(raw.get("multiClassIds")),
            keywordIds=_int_list(raw.get("keywordIds")),
            childIds=_int_list(raw.get("childIds")),
            artistName=str(raw.get("artistName", "")).strip(),
            slug=str(raw.get("slug", "")).strip(),
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "text": self.text,
            "manaCost": self.manaCost,
            "attack": self.attack,
            "health": self.health,
            "cardTypeId": self.cardTypeId,
            "classId": self.classId,
            "spellSchoolId": self.spellSchoolId,
            "minionTypeId": self.minionTypeId,
            "keywordIds": self.keywordIds,
        }

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KgRunConfig:
    source_jsonl: str
    out_dir: str
    metadata_json: str | None
    prompt_template: str
    limit: int | None
    sample_size: int | None
    random_seed: int
    chunk_size: int
    dry_run: bool
    provider: str
    model: str
    temperature: float
    timeout_seconds: int
    resume: bool
    force_llm: bool
    visualize: bool


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int)]
