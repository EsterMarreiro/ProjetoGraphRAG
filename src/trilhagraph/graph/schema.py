"""Definicao idempotente do schema LPG do TrilhaGraph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ENTITY_LABELS = (
    "Area",
    "Curso",
    "Competencia",
    "Perfil",
    "Nivel",
    "Trilha",
    "Documento",
    "Trecho",
)

NAMED_LABELS = (
    "Area",
    "Curso",
    "Competencia",
    "Perfil",
    "Nivel",
    "Trilha",
)


@dataclass(frozen=True, slots=True)
class SchemaReport:
    indices_created: int = 0
    indices_existing: int = 0
    constraints_created: int = 0
    constraints_existing: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "indices_created": self.indices_created,
            "indices_existing": self.indices_existing,
            "constraints_created": self.constraints_created,
            "constraints_existing": self.constraints_existing,
        }


class SchemaManager:
    """Cria indices e constraints sem falhar quando eles ja existem."""

    def __init__(self, graph: Any, *, embedding_dimension: int = 768) -> None:
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension deve ser maior que zero")
        self.graph = graph
        self.embedding_dimension = embedding_dimension

    def ensure(self) -> SchemaReport:
        created_indices = 0
        existing_indices = 0

        index_operations: list[Callable[[], Any]] = []
        index_operations.extend(
            lambda label=label: self.graph.create_node_range_index(label, "id")
            for label in ENTITY_LABELS
        )
        index_operations.extend(
            (
                lambda: self.graph.create_node_range_index("Area", "nome"),
                lambda: self.graph.create_node_range_index("Curso", "codigo"),
                lambda: self.graph.create_node_range_index("Area", "tipo"),
                lambda: self.graph.create_node_range_index("Curso", "nome"),
                lambda: self.graph.create_node_range_index("Curso", "tipo"),
                lambda: self.graph.create_node_range_index("Competencia", "nome"),
                lambda: self.graph.create_node_fulltext_index(
                    "Area", "nome", "descricao"
                ),
                lambda: self.graph.create_node_fulltext_index(
                    "Curso", "nome", "descricao"
                ),
                lambda: self.graph.create_node_fulltext_index(
                    "Competencia", "nome", "descricao"
                ),
                lambda: self.graph.create_node_fulltext_index("Trecho", "texto"),
                lambda: self.graph.create_node_vector_index(
                    "Trecho",
                    "embedding",
                    dim=self.embedding_dimension,
                    similarity_function="cosine",
                ),
                lambda: self.graph.create_node_vector_index(
                    "Area",
                    "embedding",
                    dim=self.embedding_dimension,
                    similarity_function="cosine",
                ),
                lambda: self.graph.create_node_vector_index(
                    "Curso",
                    "embedding",
                    dim=self.embedding_dimension,
                    similarity_function="cosine",
                ),
            )
        )

        for operation in index_operations:
            if self._create_index(operation):
                created_indices += 1
            else:
                existing_indices += 1

        created_constraints, existing_constraints = self._ensure_constraints()
        return SchemaReport(
            indices_created=created_indices,
            indices_existing=existing_indices,
            constraints_created=created_constraints,
            constraints_existing=existing_constraints,
        )

    @staticmethod
    def _is_existing_index_error(exc: Exception) -> bool:
        message = str(exc).casefold()
        markers = (
            "already indexed",
            "already exists",
            "index already exists",
            "duplicate index",
        )
        return any(marker in message for marker in markers)

    def _create_index(self, operation: Callable[[], Any]) -> bool:
        try:
            operation()
        except Exception as exc:
            if self._is_existing_index_error(exc):
                return False
            raise
        return True

    def _ensure_constraints(self) -> tuple[int, int]:
        current = {
            self._constraint_key(
                str(item["type"]),
                str(item["label"]),
                tuple(str(prop) for prop in item["properties"]),
                str(item["entitytype"]),
            )
            for item in self.graph.list_constraints()
        }

        specifications: list[tuple[str, str, tuple[str, ...]]] = []
        specifications.extend(("UNIQUE", label, ("id",)) for label in ENTITY_LABELS)
        specifications.extend(("MANDATORY", label, ("id",)) for label in ENTITY_LABELS)
        specifications.extend(("MANDATORY", label, ("nome",)) for label in NAMED_LABELS)
        specifications.extend(
            (
                ("MANDATORY", "Documento", ("arquivo",)),
                ("MANDATORY", "Documento", ("checksum",)),
                ("MANDATORY", "Trecho", ("texto",)),
            )
        )

        created = 0
        existing = 0
        for constraint_type, label, properties in specifications:
            key = self._constraint_key(constraint_type, label, properties, "NODE")
            if key in current:
                existing += 1
                continue

            if constraint_type == "UNIQUE":
                self.graph.create_node_unique_constraint(label, *properties)
            else:
                self.graph.create_node_mandatory_constraint(label, *properties)
            current.add(key)
            created += 1

        return created, existing

    @staticmethod
    def _constraint_key(
        constraint_type: str,
        label: str,
        properties: tuple[str, ...],
        entity_type: str,
    ) -> tuple[str, str, tuple[str, ...], str]:
        return (
            constraint_type.upper(),
            label,
            tuple(properties),
            entity_type.upper(),
        )
