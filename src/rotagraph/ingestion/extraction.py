"""Extracao estruturada de entidades e relacoes turisticas."""

from __future__ import annotations

from typing import Any

from rotagraph.ingestion.models import (
    ALLOWED_LABELS,
    ALLOWED_RELATIONSHIPS,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from rotagraph.ingestion.ollama import OllamaClient

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["ref", "label", "name", "properties"],
                "properties": {
                    "ref": {"type": "string"},
                    "label": {"type": "string", "enum": sorted(ALLOWED_LABELS)},
                    "name": {"type": "string"},
                    "properties": {"type": "object"},
                },
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "source_ref",
                    "target_ref",
                    "relationship_type",
                    "confidence",
                    "properties",
                ],
                "properties": {
                    "source_ref": {"type": "string"},
                    "target_ref": {"type": "string"},
                    "relationship_type": {
                        "type": "string",
                        "enum": sorted(ALLOWED_RELATIONSHIPS),
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "properties": {"type": "object"},
                },
            },
        },
    },
    "required": ["entities", "relationships"],
}

SYSTEM_PROMPT = """Voce extrai um grafo de conhecimento de turismo brasileiro.
Use somente fatos explicitamente presentes no trecho. Nao complete conhecimento
externo. Cada entidade deve ter uma ref local unica. Relacoes so podem apontar para
refs declaradas. Preserve nomes proprios e use siglas oficiais de UF quando o texto
as fornecer. Se o trecho nao sustentar uma entidade ou relacao, nao a inclua."""


class TourismExtractor:
    def __init__(self, client: OllamaClient, *, model: str) -> None:
        self.client = client
        self.model = model

    def extract(self, text: str) -> ExtractionResult:
        raw = self.client.chat_json(
            model=self.model,
            system=SYSTEM_PROMPT,
            prompt=f"Extraia entidades e relacoes deste trecho:\n\n{text}",
            schema=EXTRACTION_SCHEMA,
        )
        return parse_extraction(raw)


def parse_extraction(raw: dict[str, Any]) -> ExtractionResult:
    entities: list[ExtractedEntity] = []
    refs: set[str] = set()
    for item in raw.get("entities", []):
        ref = _required_string(item, "ref")
        label = _required_string(item, "label")
        name = _required_string(item, "name")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Label nao permitido: {label}")
        if ref in refs:
            raise ValueError(f"Referencia de entidade duplicada: {ref}")
        properties = item.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("properties de entidade deve ser um objeto")
        entities.append(ExtractedEntity(ref, label, name, properties))
        refs.add(ref)

    relationships: list[ExtractedRelationship] = []
    for item in raw.get("relationships", []):
        source = _required_string(item, "source_ref")
        target = _required_string(item, "target_ref")
        relationship_type = _required_string(item, "relationship_type")
        if source not in refs or target not in refs:
            raise ValueError("Relacao aponta para referencia inexistente")
        if relationship_type not in ALLOWED_RELATIONSHIPS:
            raise ValueError(f"Relacionamento nao permitido: {relationship_type}")
        confidence = float(item.get("confidence", 1.0))
        if not 0 <= confidence <= 1:
            raise ValueError("confidence deve estar entre 0 e 1")
        properties = item.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("properties de relacao deve ser um objeto")
        relationships.append(
            ExtractedRelationship(
                source,
                target,
                relationship_type,
                confidence,
                properties,
            )
        )
    return ExtractionResult(entities, relationships)


def _required_string(item: Any, key: str) -> str:
    if not isinstance(item, dict):
        raise ValueError("Item extraido deve ser um objeto")
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Campo obrigatorio invalido: {key}")
    return value.strip()
