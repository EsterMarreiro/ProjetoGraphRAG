"""Orquestracao da recuperacao hibrida e multi-hop."""

from __future__ import annotations

from licenciagraph.ingestion.llm import OpenAIClient
from licenciagraph.retrieval.fusion import reciprocal_rank_fusion
from licenciagraph.retrieval.models import GraphNode, RetrievalResult
from licenciagraph.retrieval.planning import CypherPlanner
from licenciagraph.retrieval.repository import (
    FULLTEXT_QUERY,
    OUTGOING_QUERY,
    SEED_QUERY,
    SOURCE_QUERY,
    VECTOR_QUERY,
    RetrievalRepository,
)

PATH_QUERY = (
    "MATCH (a {id: $source}), (b {id: $target}) "
    "RETURN shortestPath((a)-[:PRECEDE*..6]-(b))"
)
COMMUNITY_LIMIT = 8
MAX_EDGES = 80
EXTRA_SOURCE_CHARS = 30000


class HybridRetriever:
    def __init__(
        self,
        *,
        repository: RetrievalRepository,
        llm: OpenAIClient,
        embedding_model: str,
        embedding_dimension: int,
        candidate_limit: int = 8,
        result_limit: int = 5,
        planner: CypherPlanner | None = None,
        mode: str = "hybrid",
    ) -> None:
        self.repository = repository
        self.llm = llm
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.candidate_limit = candidate_limit
        self.result_limit = result_limit
        self.planner = planner
        if mode not in {"hybrid", "vector"}:
            raise ValueError("Modo deve ser hybrid ou vector")
        self.mode = mode

    def retrieve(self, question: str) -> RetrievalResult:
        warnings = getattr(self.repository, "warnings", None)
        if isinstance(warnings, list):
            warnings.clear()
        question = question.strip()
        if not question:
            raise ValueError("A pergunta nao pode ser vazia")
        embedding = self.llm.embed(
            model=self.embedding_model,
            texts=[question],
            dimensions=self.embedding_dimension,
        )[0]
        vector_hits = self.repository.vector_search(
            embedding, limit=self.candidate_limit
        )
        if self.mode == "vector":
            hits = vector_hits[: self.result_limit]
            return RetrievalResult(
                question=question,
                hits=hits,
                nodes=[],
                edges=[],
                queries=[VECTOR_QUERY],
                confidence=_retrieval_confidence(hits, 0),
            )
        fulltext_hits = self.repository.fulltext_search(
            question, limit=self.candidate_limit
        )
        channels = {"vector": vector_hits, "fulltext": fulltext_hits}
        weights = {"vector": 0.55, "fulltext": 0.45}
        queries = [VECTOR_QUERY, FULLTEXT_QUERY, SEED_QUERY]
        if self.planner:
            plan = self.planner.plan(question)
            if plan:
                channels["cypher"] = self.repository.structured_search(plan)
                weights = {"vector": 0.4, "fulltext": 0.35, "cypher": 0.25}
                queries.extend([plan.cypher, SOURCE_QUERY])
        hits = reciprocal_rank_fusion(
            channels,
            weights=weights,
            limit=self.result_limit,
            reserve_top=1,
        )
        linked = self.repository.linked_entities(question)
        similar = (
            []
            if any(node.label == "Atividade" for node in linked)
            else self.repository.similar_activities(embedding)
        )
        path_ids = self._modality_paths(linked)
        nodes, edges = self.repository.expand(
            [hit.chunk_id for hit in hits],
            max_hops=3,
            seeds=[*linked, *similar],
            seed_ids=[node_id for path in path_ids for node_id in path],
        )
        known = {hit.chunk_id for hit in hits}
        edges = _unique_edges(edges, known)[:MAX_EDGES]
        missing = list(
            dict.fromkeys(e.chunk_id for e in edges if e.chunk_id not in known)
        )
        if missing:
            extra = self.repository.sources(missing)
            budget = EXTRA_SOURCE_CHARS
            for hit in extra:
                if budget - len(hit.text) < 0 and budget < EXTRA_SOURCE_CHARS:
                    continue
                budget -= len(hit.text)
                hits.append(hit)
            queries.append(SOURCE_QUERY)
        # Todo fato do grafo entregue ao gerador precisa de uma fonte F numerada.
        known = {hit.chunk_id for hit in hits}
        edges = [edge for edge in edges if edge.chunk_id in known]
        names = {node.id: node for node in nodes}
        paths = [
            [names[i].name for i in path if i in names] for path in path_ids if path
        ]
        queries.append(OUTGOING_QUERY)
        if paths:
            queries.append(PATH_QUERY)
        return RetrievalResult(
            question=question,
            hits=hits,
            nodes=nodes,
            edges=edges,
            queries=queries,
            confidence=_retrieval_confidence(hits, len(edges)),
            warnings=list(warnings) if isinstance(warnings, list) else [],
            paths=paths,
            communities=self._activity_communities(nodes),
            similar=[node.name for node in similar],
            linked_activities=[n.name for n in linked if n.label == "Atividade"],
        )

    def _modality_paths(self, linked: list[GraphNode]) -> list[list[str]]:
        modalities = [node for node in linked if node.label == "Modalidade"]
        paths = []
        for index, first in enumerate(modalities):
            for second in modalities[index + 1 :]:
                path = self.repository.precedence_path(first.id, second.id)
                if len(path) > 2:
                    paths.append(path)
        return paths[:3]

    def _activity_communities(self, nodes: list[GraphNode]) -> list[list[str]]:
        activities = [node.id for node in nodes if node.label == "Atividade"][:3]
        if not activities:
            return []
        groups = []
        for community in self.repository.activity_communities():
            ids = [member.id for member in community]
            if any(activity in ids for activity in activities) and len(ids) > 1:
                groups.append([member.name for member in community][:COMMUNITY_LIMIT])
        return groups[:3]


def _unique_edges(edges, known):
    """Uma aresta por (origem, tipo, destino), preferindo fontes já recuperadas."""
    chosen: dict[tuple[str, str, str], object] = {}
    for edge in edges:
        key = (edge.source, edge.relationship_type, edge.target)
        current = chosen.get(key)
        if current is None or (
            current.chunk_id not in known and edge.chunk_id in known
        ):
            chosen[key] = edge
    return list(chosen.values())


def _retrieval_confidence(hits, edge_count: int) -> float:
    if not hits:
        return 0.0
    top_score = hits[0].score
    channel_coverage = max(len(hit.channels) for hit in hits) / 2
    graph_signal = min(edge_count / 5, 1.0)
    confidence = 0.15 + 0.5 * top_score + 0.2 * channel_coverage + 0.15 * graph_signal
    return round(min(max(confidence, 0.0), 0.99), 3)
