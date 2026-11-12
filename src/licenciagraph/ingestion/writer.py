"""Persistencia do resultado Text-to-Graph no FalkorDB."""

from __future__ import annotations

from typing import Any

from licenciagraph.ingestion.models import (
    ALLOWED_LABELS,
    ALLOWED_RELATIONSHIPS,
    Chunk,
    Document,
    ExtractedEntity,
    ExtractedRelationship,
)

ENTITY_PROPERTY_KEYS = frozenset(
    {
        "descricao",
        "tipo",
        "codigo",
        "sigla",
        "aliases",
        "grupo",
        "subgrupo",
        "familia",
        "modalidade",
        "paginas",
        "categoria",
        "condicao",
        "criterio",
        "letra",
        "classe",
        "ar",
        "agua",
        "solo",
        "anotacao",
        "porte",
        "unidade_porte",
    }
)
RELATIONSHIP_PROPERTY_KEYS = frozenset(
    {"ordem", "observacao", "validade", "letra", "valor_ufr"}
)


class GraphWriter:
    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self._known: set[tuple[str, str]] = set()

    def _node_clause(
        self, alias: str, label: str, node_id: str, required: tuple[str, ...]
    ) -> str:
        # Some engine versions enforce mandatory properties at CREATE, before
        # a following SET. Include them in the pattern when creating a node.
        exists = (label, node_id) in self._known or self.graph.ro_query(
            f"MATCH (n:{label} {{id: $id}}) RETURN count(n)",
            params={"id": node_id},
        ).result_set[0][0]
        self._known.add((label, node_id))
        if exists:
            return f"MATCH ({alias}:{label} {{id: $id}})"
        properties = ", ".join(f"{key}: ${key}" for key in ("id", *required))
        return f"CREATE ({alias}:{label} {{{properties}}})"

    def is_document_current(self, document: Document) -> bool:
        result = self.graph.ro_query(
            "MATCH (d:Documento {id: $id, checksum: $checksum, "
            "ingestao_completa: true}) RETURN count(d)",
            params={"id": document.id, "checksum": document.checksum},
        ).result_set
        return bool(result and result[0][0] == 1)

    def prepare_document(self, document: Document) -> None:
        # Invalidate completion before any destructive work. A failed load must
        # be retried, including when the checksum has not changed.
        self.graph.query(
            "MATCH (d:Documento {id: $id}) SET d.ingestao_completa = false",
            params={"id": document.id},
        )
        self.graph.query(
            """
            MATCH (d:Documento {id: $id})-[:CONTEM]->(t:Trecho)
            MATCH ()-[r]->() WHERE r.trecho_id = t.id DELETE r
            """,
            params={"id": document.id},
        )
        self.graph.query(
            "MATCH (d:Documento {id: $id})-[:CONTEM]->(t:Trecho) DETACH DELETE t",
            params={"id": document.id},
        )
        clause = self._node_clause(
            "d", "Documento", document.id, ("arquivo", "checksum")
        )
        self.graph.query(
            clause
            + """
            SET d.titulo = $titulo,
                d.arquivo = $arquivo,
                d.checksum = $checksum,
                d.ingestao_completa = false
            """,
            params={
                "id": document.id,
                "titulo": document.title,
                "arquivo": document.path.name,
                "checksum": document.checksum,
            },
        )

    def remove_orphans(self) -> int:
        """Remove entidades de domínio sem nenhuma relação (restos de reingestões)."""
        result = self.graph.query(
            "MATCH (n) WHERE NOT n:Trecho AND NOT n:Documento AND NOT (n)--() "
            "DELETE n RETURN count(n)"
        ).result_set
        self._known.clear()
        return int(result[0][0]) if result else 0

    def complete_document(self, document: Document) -> None:
        self.graph.query(
            "MATCH (d:Documento {id: $id, checksum: $checksum}) "
            "SET d.ingestao_completa = true",
            params={"id": document.id, "checksum": document.checksum},
        )

    def upsert_chunk(self, chunk: Chunk, embedding: list[float]) -> None:
        self._known.discard(("Trecho", chunk.id))
        clause = self._node_clause("t", "Trecho", chunk.id, ("texto",))
        self.graph.query(
            """
            MATCH (d:Documento {id: $document_id})
            """
            + clause
            + """
            SET t.texto = $texto,
                t.ordem = $ordem,
                t.pagina = $pagina,
                t.tipo = $tipo,
                t.embedding = vecf32($embedding)
            MERGE (d)-[:CONTEM]->(t)
            """,
            params={
                "document_id": chunk.document_id,
                "id": chunk.id,
                "texto": chunk.text,
                "ordem": chunk.position,
                "pagina": chunk.page if chunk.page is not None else 0,
                "tipo": chunk.kind,
                "embedding": embedding,
            },
        )

    def upsert_entity(
        self, entity: ExtractedEntity, embedding: list[float] | None
    ) -> None:
        if entity.label not in ALLOWED_LABELS or entity.canonical_id is None:
            raise ValueError("Entidade nao validada ou nao canonicalizada")

        params: dict[str, Any] = {"id": entity.canonical_id, "nome": entity.name}
        assignments = ["e.nome = $nome"]
        if embedding is not None:
            params["embedding"] = embedding
            assignments.append("e.embedding = vecf32($embedding)")
        for key, value in entity.properties.items():
            if key in ENTITY_PROPERTY_KEYS and value is not None:
                params[key] = value
                assignments.append(f"e.{key} = ${key}")

        query = (
            self._node_clause("e", entity.label, entity.canonical_id, ("nome",))
            + " SET "
            + ", ".join(assignments)
        )
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
            MERGE (source)-[r:{relationship.relationship_type}
                {{trecho_id: $trecho_id}}]->(target)
            SET {", ".join(assignments)}
        """
        self.graph.query(query, params=params)
