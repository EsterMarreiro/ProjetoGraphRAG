"""Registros curados extraídos das fontes oficiais."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ChecklistItem:
    number: int
    text: str


@dataclass(slots=True)
class Checklist:
    code: str
    family: str
    modality: str
    heading: list[str]
    first_page: int
    last_page: int
    variant: str = ""
    items: list[ChecklistItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    annex: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Activity:
    code: str
    name: str
    group_code: str
    group_name: str
    subgroup_code: str
    subgroup_name: str
    regime: str
    checklists: list[str]
    page: int
    by_modality: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ManualData:
    activities: list[Activity]
    checklists: list[Checklist]
    source: str = "manual-licenciamento-5a-edicao"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
