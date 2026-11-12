"""Modelos internos da ingestao."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_LABELS = frozenset(
    {
        "Grupo",
        "Subgrupo",
        "Atividade",
        "Modalidade",
        "Checklist",
        "DocumentoExigido",
        "Estudo",
        "Orgao",
        "Sistema",
        "Norma",
        "Porte",
        "Faixa",
        "LetraCusto",
    }
)

RELATIONSHIP_ENDPOINTS = {
    "SUBGRUPO_DE": ("Subgrupo", "Grupo"),
    "CLASSIFICADA_EM": ("Atividade", "Subgrupo"),
    "ENQUADRADA_EM": ("Atividade", "Modalidade"),
    "EXIGE_CHECKLIST": ("Atividade", "Checklist"),
    "DO_TIPO": ("Checklist", "Modalidade"),
    "EXIGE_DOCUMENTO": ("Checklist", "DocumentoExigido"),
    "EMITIDO_POR": ("DocumentoExigido", "Orgao"),
    "EXIGE_ESTUDO": ("Checklist", "Estudo"),
    "TEM_FAIXA": ("Atividade", "Faixa"),
    "DE_PORTE": ("Faixa", "Porte"),
    "USA_LETRA": ("Faixa", "LetraCusto"),
    "ESTUDO_DA_FAIXA": ("Faixa", "Estudo"),
    "CUSTA": ("LetraCusto", "Modalidade"),
    "PRECEDE": ("Modalidade", "Modalidade"),
    "REGULADO_POR": ("Modalidade", "Norma"),
    "EMITIDA_POR": ("Modalidade", "Orgao"),
    "REQUER_ESTUDO": ("Modalidade", "Estudo"),
    "OPERA": ("Orgao", "Sistema"),
    "VINCULADO_A": ("Orgao", "Orgao"),
}
ALLOWED_RELATIONSHIPS = frozenset(RELATIONSHIP_ENDPOINTS)


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
    page: int | None = None
    kind: str = "texto"


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
