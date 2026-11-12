"""Extração estruturada de cursos, competências e pré-requisitos."""

from __future__ import annotations

from typing import Any

from licenciagraph.ingestion.catalog import (
    CATALOG_PROMPT,
    CATALOG_SCHEMA,
    parse_catalog,
)
from licenciagraph.ingestion.llm import OpenAIClient
from licenciagraph.ingestion.models import (
    ALLOWED_LABELS,
    ALLOWED_RELATIONSHIPS,
    RELATIONSHIP_ENDPOINTS,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)

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
                    "properties": {
                        "type": "object",
                        "properties": {
                            "descricao": {"type": "string"},
                            "tipo": {"type": "string"},
                            "codigo": {"type": "string"},
                            "carga_horaria": {"type": "integer"},
                            "aliases": {"type": "array", "items": {"type": "string"}},
                        },
                        "additionalProperties": False,
                    },
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
                    "properties": {
                        "type": "object",
                        "properties": {
                            "ordem": {"type": "integer"},
                            "observacao": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
        },
    },
    "required": ["entities", "relationships"],
}

SYSTEM_PROMPT = """Você extrai um grafo de um catálogo didático fictício de cursos.
Use somente fatos explícitos. Curso representa disciplinas; Competencia representa
habilidades; Area representa áreas de conhecimento; Perfil representa públicos;
Nivel representa Iniciante, Intermediário ou Avançado; Trilha representa percursos.
Curso ABORDA Area; Curso DESENVOLVE Competencia; Curso INDICADO_PARA Perfil;
Curso CLASSIFICADO_EM Nivel; Trilha INCLUI Curso; Curso PRE_REQUISITO Curso:
a direção é do curso avançado para o curso exigido antes dele. Extraia a relação
PRE_REQUISITO apenas se o texto disser que o primeiro exige o segundo.
Cada entidade tem ref única. Preserve nomes completos e acentos. Não extraia
metadados, datas ou frases inteiras como entidades. Não invente propriedades.
As propriedades podem conter descricao, tipo, codigo, carga_horaria e aliases.
Não introduza provedor ou edição se não houver desambiguação explícita.
Só use refs declaradas em entities nas relações. Liste todas as relações explícitas.
Se o texto disser 'A trilha T inclui o curso C', INCLUI sai de T (Trilha)
e chega em C (Curso), nunca o contrário. Use properties={} quando não houver
atributos explícitos. codigo e carga_horaria pertencem apenas a Curso.
"""


class LearningExtractor:
    def __init__(self, client: OpenAIClient, *, model: str) -> None:
        self.client = client
        self.model = model

    def extract(self, text: str) -> ExtractionResult:
        feedback = ""
        for attempt in range(3):
            raw = self.client.chat_json(
                model=self.model,
                system=CATALOG_PROMPT,
                prompt=(
                    f"Extraia entidades e relacoes deste trecho:\n\n{text}{feedback}"
                ),
                schema=CATALOG_SCHEMA,
            )
            try:
                return parse_catalog(raw, text)
            except ValueError as exc:
                if attempt == 2:
                    raise
                feedback = (
                    f"\n\nA extração anterior foi rejeitada: {exc}. "
                    "Releia o trecho e corrija o esquema sem inventar fatos."
                )
        raise AssertionError("Fluxo de extração inesperado")


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
    labels_by_ref = {entity.ref: entity.label for entity in entities}
    for item in raw.get("relationships", []):
        source = _required_string(item, "source_ref")
        target = _required_string(item, "target_ref")
        relationship_type = _required_string(item, "relationship_type")
        if source not in refs or target not in refs:
            raise ValueError("Relacao aponta para referencia inexistente")
        if relationship_type not in ALLOWED_RELATIONSHIPS:
            raise ValueError(f"Relacionamento nao permitido: {relationship_type}")
        pair = (labels_by_ref[source], labels_by_ref[target])
        required = RELATIONSHIP_ENDPOINTS[relationship_type]
        if pair != required:
            raise ValueError(
                f"{relationship_type} exige {required[0]} -> {required[1]}; "
                f"recebido {pair[0]} -> {pair[1]}"
            )
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
