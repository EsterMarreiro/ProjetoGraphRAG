"""Consultas Cypher somente-leitura usadas pela recuperacao."""

from __future__ import annotations

import re
from typing import Any

from licenciagraph.ingestion.canonicalization import normalize_text
from licenciagraph.retrieval.models import GraphEdge, GraphNode, SearchHit
from licenciagraph.retrieval.planning import CYPHER_CHOICES, CypherPlan

SOURCE_QUERY = """
UNWIND $chunk_ids AS chunk_id
MATCH (document:Documento)-[:CONTEM]->(node:Trecho {id: chunk_id})
WHERE document.ingestao_completa = true
RETURN node.id, node.texto, document.id, document.titulo, document.arquivo, 1.0
"""

VECTOR_QUERY = """
CALL db.idx.vector.queryNodes(
    'Trecho', 'embedding', $limit, vecf32($embedding)
) YIELD node, score
MATCH (document:Documento)-[:CONTEM]->(node)
WHERE document.ingestao_completa = true
RETURN node.id, node.texto, document.id, document.titulo, document.arquivo,
       (1.0 - score) AS similarity
ORDER BY similarity DESC
"""

FULLTEXT_QUERY = """
CALL db.idx.fulltext.queryNodes('Trecho', $query) YIELD node, score
MATCH (document:Documento)-[:CONTEM]->(node)
WHERE document.ingestao_completa = true
RETURN node.id, node.texto, document.id, document.titulo, document.arquivo, score
ORDER BY score DESC
LIMIT $limit
"""

SEED_QUERY = """
UNWIND $chunk_ids AS chunk_id
MATCH (chunk:Trecho {id: chunk_id})-[:MENCIONA]->(entity)
RETURN DISTINCT entity.id, labels(entity)[0], entity.nome
"""

# Nós-hub (Area, Nivel, Perfil, Trilha) aparecem como vizinhos, mas não são expandidos.
# A segunda fronteira segue apenas pré-requisitos e competências, sem espalhar o
# contexto pelo catálogo inteiro.
NEIGHBOR_QUERY = """
UNWIND $entity_ids AS entity_id
MATCH (focus {id: entity_id})-[relationship]-(other)
WHERE (focus:Curso OR focus:Competencia)
  AND NOT other:Trecho AND NOT other:Documento
RETURN DISTINCT
    focus.id, labels(focus)[0], focus.nome,
    other.id, labels(other)[0], other.nome,
    startNode(relationship).id, endNode(relationship).id,
    type(relationship), relationship.confianca, relationship.trecho_id
LIMIT $limit
"""


NAMED_HOP_QUERY = NEIGHBOR_QUERY.replace(
    "WHERE (focus:Curso OR focus:Competencia)\n  AND NOT other:Trecho",
    "WHERE NOT focus:Trecho AND NOT focus:Documento\n  AND NOT other:Trecho",
)

ENTITY_QUERY = """
MATCH (n)
WHERE NOT n:Trecho AND NOT n:Documento
RETURN n.id, labels(n)[0], n.nome
"""

SECOND_HOP_QUERY = NEIGHBOR_QUERY.replace(
    "WHERE (focus:Curso OR focus:Competencia)",
    "WHERE focus:Curso AND type(relationship) IN ['PRE_REQUISITO', 'DESENVOLVE']",
)


class RetrievalRepository:
    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.warnings: list[str] = []
        self._entities: list[GraphNode] | None = None

    def structured_search(self, plan: CypherPlan) -> list[SearchHit]:
        if plan.cypher not in CYPHER_CHOICES:
            raise ValueError("Consulta fora da gramática permitida")
        try:
            rows = self.graph.ro_query(
                plan.cypher, params={"name": plan.name}, timeout=10000
            ).result_set
        except Exception as exc:
            if "query timed out" not in str(exc).lower():
                raise
            self.warnings.append(
                "A consulta Cypher gerada excedeu o prazo; "
                "a recuperação prosseguiu pelos demais canais."
            )
            return []
        return self.sources([str(row[0]) for row in rows if row[0]], "cypher")

    def sources(self, chunk_ids: list[str], channel: str = "graph") -> list[SearchHit]:
        if not chunk_ids:
            return []
        rows = self.graph.ro_query(
            SOURCE_QUERY, params={"chunk_ids": list(dict.fromkeys(chunk_ids))}
        ).result_set
        return [self._row_to_hit(row, channel) for row in rows]

    def vector_search(
        self, embedding: list[float], *, limit: int = 8
    ) -> list[SearchHit]:
        rows = self.graph.ro_query(
            VECTOR_QUERY, params={"embedding": embedding, "limit": limit}
        ).result_set
        return [self._row_to_hit(row, "vector") for row in rows]

    def fulltext_search(self, question: str, *, limit: int = 8) -> list[SearchHit]:
        query = _fulltext_query(question)
        if not query:
            return []
        rows = self.graph.ro_query(
            FULLTEXT_QUERY, params={"query": query, "limit": limit}
        ).result_set
        return [self._row_to_hit(row, "fulltext") for row in rows]

    def entities(self) -> list[GraphNode]:
        if self._entities is None:
            rows = self.graph.ro_query(ENTITY_QUERY).result_set
            self._entities = [
                GraphNode(str(row[0]), str(row[1]), str(row[2] or row[0]))
                for row in rows
            ]
        return self._entities

    def linked_entities(self, question: str) -> list[GraphNode]:
        """Entidades citadas pelo nome, ignorando nomes contidos em outros."""
        asked = normalize_text(question)
        spans = []
        for node in self.entities():
            name = normalize_text(node.name)
            if len(name) < 3:
                continue
            for match in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", asked):
                spans.append((match.start(), match.end(), node))
        kept = [
            span
            for span in spans
            if not any(
                other is not span
                and other[0] <= span[0]
                and span[1] <= other[1]
                and (other[1] - other[0]) > (span[1] - span[0])
                for other in spans
            )
        ]
        return list({span[2].id: span[2] for span in kept}.values())

    def prerequisite_path(self, source_id: str, target_id: str) -> list[str]:
        from licenciagraph.retrieval.algorithms import GraphAlgorithms

        path = GraphAlgorithms(self.graph).shortest_path(
            source_id, target_id, relationship_type="PRE_REQUISITO"
        )
        return path.node_ids if path else []

    def expand(
        self,
        chunk_ids: list[str],
        *,
        max_hops: int = 2,
        limit: int = 200,
        question: str | None = None,
        seed_ids: list[str] | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        if max_hops not in {1, 2}:
            raise ValueError("max_hops deve ser 1 ou 2")
        if not chunk_ids:
            return [], []

        named = {n.id: n for n in self.linked_entities(question)} if question else {}
        if seed_ids:
            by_id = {n.id: n for n in self.entities()}
            named.update({i: by_id[i] for i in seed_ids if i in by_id})
        if named:
            nodes = dict(named)
        else:
            seed_rows = self.graph.ro_query(
                SEED_QUERY, params={"chunk_ids": chunk_ids}
            ).result_set
            nodes = {
                str(row[0]): GraphNode(str(row[0]), str(row[1]), str(row[2] or row[0]))
                for row in seed_rows
            }
        frontier = list(nodes)
        edges: dict[tuple[str, str, str, str | None], GraphEdge] = {}

        for hop in range(max_hops):
            if not frontier:
                break
            if hop == 0:
                query = NAMED_HOP_QUERY if named else NEIGHBOR_QUERY
            else:
                query = SECOND_HOP_QUERY
            rows = self.graph.ro_query(
                query, params={"entity_ids": frontier, "limit": limit}
            ).result_set
            next_frontier: list[str] = []
            for row in rows:
                focus = GraphNode(str(row[0]), str(row[1]), str(row[2] or row[0]))
                other = GraphNode(str(row[3]), str(row[4]), str(row[5] or row[3]))
                nodes[focus.id] = focus
                if other.id not in nodes:
                    next_frontier.append(other.id)
                nodes[other.id] = other
                edge = GraphEdge(
                    source=str(row[6]),
                    target=str(row[7]),
                    relationship_type=str(row[8]),
                    confidence=float(row[9]) if row[9] is not None else None,
                    chunk_id=str(row[10]) if row[10] is not None else None,
                )
                edges[
                    (edge.source, edge.target, edge.relationship_type, edge.chunk_id)
                ] = edge
            frontier = list(dict.fromkeys(next_frontier))

        return list(nodes.values()), list(edges.values())

    @staticmethod
    def _row_to_hit(row: list[Any], channel: str) -> SearchHit:
        return SearchHit(
            chunk_id=str(row[0]),
            text=str(row[1]),
            document_id=str(row[2]),
            title=str(row[3]),
            filename=str(row[4]),
            score=float(row[5]),
            channels={channel},
        )


def _fulltext_query(question: str) -> str:
    tokens = re.findall(r"[\wÀ-ÿ]+", question, flags=re.UNICODE)
    meaningful = [token for token in tokens if len(token) >= 3]
    return " | ".join(meaningful[:20])
