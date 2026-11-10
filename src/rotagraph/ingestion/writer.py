"""Persistencia do resultado Text-to-Graph no FalkorDB."""

from __future__ import annotations

from typing import Any

from rotagraph.ingestion.models import (
    ALLOWED_LABELS,
    ALLOWED_RELATIONSHIPS,
    Chunk,
    Document,
    ExtractedEntity,
    ExtractedRelationship,
)

ENTITY_PROPERTY_KEYS = frozenset(
    {"descricao", "tipo", "uf", "municipio", "aliases", "duracao_dias"}
)
RELATIONSHIP_PROPERTY_KEYS = frozenset({"ordem", "distancia_km", "observacao"})


class GraphWriter:
    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def is_document_current(self, document: Document) -> bool:
        result = self.graph.ro_query(
            "MATCH (d:Documento {id: $id, checksum: $checksum}) RETURN count(d)",
            params={"id": document.id, "checksum": document.checksum},
        ).result_set
        return bool(result and result[0][0] == 1)

    def prepare_document(self, document: Document) -> None:
        self.graph.query(
            "MATCH (d:Documento {id: $id})-[:CONTEM]->(t:Trecho) DETACH DELETE t",
            params={"id": document.id},
        )
        self.graph.query(
            """
            MERGE (d:Documento {id: $id})
            SET d.titulo = $titulo,
                d.arquivo = $arquivo,
                d.checksum = $checksum
            """,
            params={
                "id": document.id,
                "titulo": document.title,
                "arquivo": document.path.name,
                "checksum": document.checksum,
            },
        )

    def upsert_chunk(self, chunk: Chunk, embedding: list[float]) -> None:
        self.graph.query(
            """
            MATCH (d:Documento {id: $document_id})
            MERGE (t:Trecho {id: $id})
            SET t.texto = $texto,
                t.ordem = $ordem,
                t.embedding = vecf32($embedding)
            MERGE (d)-[:CONTEM]->(t)
            """,
            params={
                "document_id": chunk.document_id,
                "id": chunk.id,
                "texto": chunk.text,
                "ordem": chunk.position,
                "embedding": embedding,
            },
        )

    def upsert_entity(self, entity: ExtractedEntity, embedding: list[float]) -> None:
        if entity.label not in ALLOWED_LABELS or entity.canonical_id is None:
            raise ValueError("Entidade nao validada ou nao canonicalizada")

        params: dict[str, Any] = {
            "id": entity.canonical_id,
            "nome": entity.name,
            "embedding": embedding,
        }
        assignments = ["e.nome = $nome", "e.embedding = vecf32($embedding)"]
        for key, value in entity.properties.items():
            if key in ENTITY_PROPERTY_KEYS and value is not None:
                params[key] = value
                assignments.append(f"e.{key} = ${key}")

        query = f"MERGE (e:{entity.label} {{id: $id}}) SET " + ", ".join(assignments)
        self.graph.query(query, params=params)

    def link_mention(self, chunk_id: str, entity: ExtractedEntity) -> None:
        if entity.label not in ALLOWED_LABELS or entity.canonical_id is None:
            raise ValueError("Entidade nao validada ou nao canonicalizada")
        self.graph.query(
            f"""
            MATCH (t:Trecho {{id: $chunk_id}})
            MATCH (e:{entity.label} {{id: $entity_id}})
            MERGE (t)-[:MENCIONA]->(e)
            """,
            params={"chunk_id": chunk_id, "entity_id": entity.canonical_id},
        )

    def upsert_relationship(
        self,
        relationship: ExtractedRelationship,
        entities_by_ref: dict[str, ExtractedEntity],
        *,
        chunk_id: str,
    ) -> None:
        if relationship.relationship_type not in ALLOWED_RELATIONSHIPS:
            raise ValueError("Tipo de relacionamento nao permitido")
        source = entities_by_ref[relationship.source_ref]
        target = entities_by_ref[relationship.target_ref]
        if source.canonical_id is None or target.canonical_id is None:
            raise ValueError("Relacao contem entidade nao canonicalizada")

        params: dict[str, Any] = {
            "source_id": source.canonical_id,
            "target_id": target.canonical_id,
            "confianca": relationship.confidence,
            "trecho_id": chunk_id,
            "metodo": "llm",
        }
        assignments = [
            "r.confianca = $confianca",
            "r.trecho_id = $trecho_id",
            "r.metodo = $metodo",
        ]
        for key, value in relationship.properties.items():
            if key in RELATIONSHIP_PROPERTY_KEYS and value is not None:
                params[key] = value
                assignments.append(f"r.{key} = ${key}")

        query = f"""
            MATCH (source:{source.label} {{id: $source_id}})
            MATCH (target:{target.label} {{id: $target_id}})
            MERGE (source)-[r:{relationship.relationship_type}]->(target)
            SET {", ".join(assignments)}
        """
        self.graph.query(query, params=params)
