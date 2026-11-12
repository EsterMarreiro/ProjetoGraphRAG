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
RETURN node.id, node.texto, document.id, document.titulo, document.arquivo, 1.0,
       node.pagina
"""

VECTOR_QUERY = """
CALL db.idx.vector.queryNodes(
    'Trecho', 'embedding', $limit, vecf32($embedding)
) YIELD node, score
MATCH (document:Documento)-[:CONTEM]->(node)
WHERE document.ingestao_completa = true
RETURN node.id, node.texto, document.id, document.titulo, document.arquivo,
       (1.0 - score) AS similarity, node.pagina
ORDER BY similarity DESC
"""

FULLTEXT_QUERY = """
CALL db.idx.fulltext.queryNodes('Trecho', $query) YIELD node, score
MATCH (document:Documento)-[:CONTEM]->(node)
WHERE document.ingestao_completa = true
RETURN node.id, node.texto, document.id, document.titulo, document.arquivo, score,
       node.pagina
ORDER BY score DESC
LIMIT $limit
"""

SEED_QUERY = """
UNWIND $chunk_ids AS chunk_id
MATCH (chunk:Trecho {id: chunk_id})-[:MENCIONA]->(entity)
RETURN DISTINCT entity.id, labels(entity)[0], entity.nome
"""

# A expansão segue só relações de saída: no grafo, as arestas vão do específico
# (atividade, checklist, faixa) para o geral (modalidade, órgão, porte), então os
# nós-hub nunca despejam centenas de vizinhos. Filtros opcionais restringem os
# checklists e as faixas à modalidade e ao porte citados na pergunta.
EDGE_COLUMNS = """
RETURN DISTINCT
    focus.id, labels(focus)[0], focus.nome,
    other.id, labels(other)[0], other.nome,
    startNode(relationship).id, endNode(relationship).id,
    type(relationship), relationship.confianca, relationship.trecho_id,
    properties(relationship)
LIMIT $limit
"""
OUTGOING_QUERY = (
    """
UNWIND $entity_ids AS entity_id
MATCH (focus:{label} {{id: entity_id}})-[relationship]->(other)
WHERE NOT other:Trecho AND NOT other:Documento
  AND (NOT other:Checklist OR size($modalities) = 0
       OR other.modalidade IN $modalities)
  AND (NOT other:Faixa OR size($portes) = 0 OR other.porte IN $portes)
  AND (NOT {label_is_letter} OR size($modality_ids) = 0
       OR other.id IN $modality_ids)
"""
    + EDGE_COLUMNS
)
INCOMING_QUERY = (
    """
UNWIND $entity_ids AS entity_id
MATCH (other)-[relationship]->(focus:{label} {{id: entity_id}})
WHERE (other:Atividade OR other:Checklist)
  AND (NOT other:Checklist OR size($modalities) = 0
       OR other.modalidade IN $modalities)
"""
    + EDGE_COLUMNS
)
HOP_SOURCES = (None, {"Checklist", "Faixa"}, {"DocumentoExigido", "LetraCusto"})
EXPANDABLE_IN = (
    "Grupo",
    "Subgrupo",
    "Modalidade",
    "DocumentoExigido",
    "Estudo",
    "Orgao",
)


def query_for_label(template: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z]+", label):
        raise ValueError("rótulo inválido")
    return template.format(
        label=label, label_is_letter="true" if label == "LetraCusto" else "false"
    )


ENTITY_QUERY = """
MATCH (n)
WHERE NOT n:Trecho AND NOT n:Documento AND NOT n:Faixa
RETURN n.id, labels(n)[0], n.nome, n.sigla, n.codigo
"""
ACTIVITY_VECTOR_QUERY = """
CALL db.idx.vector.queryNodes(
    'Atividade', 'embedding', $limit, vecf32($embedding)
) YIELD node, score
RETURN node.id, 'Atividade', node.nome, (1.0 - score) AS similarity
ORDER BY similarity DESC
"""
LISTING_QUERY = """
MATCH (activity:Atividade)-[relationship:ENQUADRADA_EM]->(modality:Modalidade)
WHERE modality.id IN $modality_ids
  AND (activity.grupo IN $group_names OR activity.subgrupo IN $subgroup_names)
RETURN DISTINCT
    activity.id AS focus_id, 'Atividade' AS focus_label, activity.nome AS focus_name,
    modality.id AS other_id, 'Modalidade' AS other_label,
    modality.nome AS other_name,
    activity.id AS source_id, modality.id AS target_id,
    type(relationship) AS relation, relationship.confianca AS confidence,
    relationship.trecho_id AS chunk, properties(relationship) AS props
LIMIT $limit
"""
ACTIVITY_PAIRS_QUERY = """
MATCH (a:Atividade)-[:EXIGE_CHECKLIST]->(c:Checklist)<-[:EXIGE_CHECKLIST]-(b:Atividade)
WHERE a.id < b.id
RETURN a.id, b.id, count(c)
"""


class RetrievalRepository:
    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.warnings: list[str] = []
        self._entities: list[GraphNode] | None = None
        self._communities: list[list[GraphNode]] | None = None

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
            self._entities = [_node_from_entity_row(row) for row in rows]
        return self._entities

    def linked_entities(self, question: str) -> list[GraphNode]:
        """Entidades citadas pelo nome, ignorando nomes contidos em outros."""
        asked = normalize_text(question)
        spans = []
        for node in self.entities():
            for alias in node.aliases:
                if alias.isupper() and len(alias) < 6:
                    pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
                    matches = re.finditer(pattern, question)
                elif alias.endswith("*"):
                    stem = re.escape(normalize_text(alias[:-1]))
                    matches = re.finditer(rf"(?<!\w){stem}\w*", asked)
                else:
                    name = normalize_text(alias)
                    if len(name) < 3:
                        continue
                    matches = re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", asked)
                for match in matches:
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

    def similar_activities(
        self, embedding: list[float], *, limit: int = 3, minimum: float = 0.52
    ) -> list[GraphNode]:
        rows = self.graph.ro_query(
            ACTIVITY_VECTOR_QUERY, params={"embedding": embedding, "limit": limit}
        ).result_set
        if not rows:
            return []
        best = float(rows[0][3])
        return [
            GraphNode(str(row[0]), "Atividade", str(row[2]))
            for row in rows
            if float(row[3]) >= minimum and float(row[3]) >= best - 0.08
        ]

    def activity_communities(self) -> list[list[GraphNode]]:
        """Comunidades de Louvain: atividades que compartilham checklists."""
        if self._communities is None:
            from licenciagraph.retrieval.algorithms import louvain_communities

            rows = self.graph.ro_query(ACTIVITY_PAIRS_QUERY).result_set
            groups = louvain_communities([(str(r[0]), str(r[1])) for r in rows])
            names = {n.id: n for n in self.entities() if n.label == "Atividade"}
            self._communities = [
                [names[i] for i in group if i in names] for group in groups
            ]
        return self._communities

    def precedence_path(self, source_id: str, target_id: str) -> list[str]:
        from licenciagraph.retrieval.algorithms import GraphAlgorithms

        path = GraphAlgorithms(self.graph).shortest_path(
            source_id, target_id, relationship_type="PRECEDE"
        )
        return path.node_ids if path else []

    def expand(
        self,
        chunk_ids: list[str],
        *,
        max_hops: int = 3,
        limit: int = 200,
        question: str | None = None,
        seed_ids: list[str] | None = None,
        seeds: list[GraphNode] | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        if not 1 <= max_hops <= 3:
            raise ValueError("max_hops deve estar entre 1 e 3")
        if not chunk_ids:
            return [], []

        named = {n.id: n for n in self.linked_entities(question)} if question else {}
        for node in seeds or []:
            named.setdefault(node.id, node)
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
        modality_ids = [n.id for n in named.values() if n.label == "Modalidade"]
        params = {
            "limit": limit,
            "modalities": [
                n.aliases[1].upper()
                for n in named.values()
                if n.label == "Modalidade" and len(n.aliases) > 1
            ],
            "portes": [
                n.id.removeprefix("porte:").upper()
                for n in named.values()
                if n.label == "Porte"
            ],
            "modality_ids": modality_ids,
        }
        frontier = list(nodes)
        edges: dict[tuple[str, str, str, str | None], GraphEdge] = {}

        def run(template: str, ids: list[str]) -> list[str]:
            by_label: dict[str, list[str]] = {}
            for node_id in ids:
                by_label.setdefault(nodes[node_id].label, []).append(node_id)
            rows = []
            for label, label_ids in by_label.items():
                rows.extend(
                    self.graph.ro_query(
                        query_for_label(template, label),
                        params={**params, "entity_ids": label_ids},
                    ).result_set
                )
            return collect(rows)

        def collect(rows: list[list[Any]]) -> list[str]:
            reached: list[str] = []
            for row in rows:
                focus = GraphNode(str(row[0]), str(row[1]), str(row[2] or row[0]))
                other = GraphNode(str(row[3]), str(row[4]), str(row[5] or row[3]))
                nodes.setdefault(focus.id, focus)
                if other.id not in nodes:
                    reached.append(other.id)
                nodes.setdefault(other.id, other)
                edge = GraphEdge(
                    source=str(row[6]),
                    target=str(row[7]),
                    relationship_type=str(row[8]),
                    confidence=float(row[9]) if row[9] is not None else None,
                    chunk_id=str(row[10]) if row[10] is not None else None,
                    properties=_edge_properties(row[11]),
                )
                edges[
                    (edge.source, edge.target, edge.relationship_type, edge.chunk_id)
                ] = edge
            return list(dict.fromkeys(reached))

        for hop in range(max_hops):
            sources = [
                node_id
                for node_id in frontier
                if HOP_SOURCES[hop] is None or nodes[node_id].label in HOP_SOURCES[hop]
            ]
            if hop == 1 and not params["modalities"]:
                sources = [i for i in sources if nodes[i].label != "Checklist"]
            if hop == 2 and not params["modality_ids"]:
                sources = [i for i in sources if nodes[i].label != "LetraCusto"]
            if not sources:
                break
            frontier = run(OUTGOING_QUERY, sources)
        incoming = [
            n.id
            for n in named.values()
            if n.label in EXPANDABLE_IN
            and not any(m.label == "Atividade" for m in named.values())
        ]
        if incoming:
            run(INCOMING_QUERY, incoming)
        groups = [n.name for n in named.values() if n.label == "Grupo"]
        subgroups = [n.name for n in named.values() if n.label == "Subgrupo"]
        if modality_ids and (groups or subgroups):
            rows = self.graph.ro_query(
                LISTING_QUERY,
                params={
                    "modality_ids": modality_ids,
                    "group_names": groups,
                    "subgroup_names": subgroups,
                    "limit": 80,
                },
            ).result_set
            collect(rows)
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
            page=int(row[6]) if len(row) > 6 and row[6] else None,
        )


def _node_from_entity_row(row: list[Any]) -> GraphNode:
    node_id, label, name, sigla, code = (
        str(row[0]),
        str(row[1]),
        str(row[2] or row[0]),
        str(row[3] or ""),
        str(row[4] or ""),
    )
    aliases = [name, re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()]
    if label == "Modalidade":
        aliases = [aliases[1], sigla]
        if sigla == "DISPENSA":
            aliases = ["dispens*"]
    elif sigla and label in {"Orgao", "Sistema", "Estudo"}:
        aliases.append(sigla)
    if code and label == "Checklist":
        aliases.append(code)
    unique = tuple(dict.fromkeys(a for a in aliases if a))
    return GraphNode(node_id, label, name, unique)


def _edge_properties(raw: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, dict):
        return ()
    ignored = {"confianca", "trecho_id", "metodo"}
    return tuple(
        (str(key), str(value))
        for key, value in sorted(raw.items())
        if key not in ignored and value not in (None, "")
    )


def _fulltext_query(question: str) -> str:
    tokens = re.findall(r"[\wÀ-ÿ]+", question, flags=re.UNICODE)
    meaningful = [token for token in tokens if len(token) >= 3]
    return " | ".join(meaningful[:20])
