"""Modelos internos da ingestao."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_LABELS = frozenset(
    {
        "Localidade",
        "Atracao",
        "Atividade",
        "Perfil",
        "Periodo",
        "Rota",
    }
)

ALLOWED_RELATIONSHIPS = frozenset(
    {
        "FAZ_PARTE_DE",
        "LOCALIZADA_EM",
        "OFERECE",
        "ADEQUADA_PARA",
        "RECOMENDADA_EM",
        "EVITAR_EM",
        "INCLUI",
        "PROXIMA_DE",
    }
)


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    path: Path
    checksum: str
    content: str


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    position: int
    text: str


@dataclass(slots=True)
class ExtractedEntity:
    ref: str
    label: str
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    canonical_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedRelationship:
    source_ref: str
    target_ref: str
    relationship_type: str
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
