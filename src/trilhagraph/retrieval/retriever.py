"""Orquestracao da recuperacao hibrida e multi-hop."""

from __future__ import annotations

from trilhagraph.ingestion.ollama import OllamaClient
from trilhagraph.retrieval.fusion import reciprocal_rank_fusion
from trilhagraph.retrieval.models import RetrievalResult
from trilhagraph.retrieval.planning import CypherPlanner
from trilhagraph.retrieval.repository import (
    FULLTEXT_QUERY,
    NEIGHBOR_QUERY,
    SEED_QUERY,
    SOURCE_QUERY,
    VECTOR_QUERY,
    RetrievalRepository,
)


class HybridRetriever:
    def __init__(
        self,
        *,
        repository: RetrievalRepository,
        ollama: OllamaClient,
        embedding_model: str,
        embedding_dimension: int,
        candidate_limit: int = 8,
        result_limit: int = 5,
        planner: CypherPlanner | None = None,
        mode: str = "hybrid",
    ) -> None:
        self.repository = repository
        self.ollama = ollama
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
        embedding = self.ollama.embed(
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
        weights = {"vector": 0.65, "fulltext": 0.35}
        queries = [VECTOR_QUERY, FULLTEXT_QUERY, SEED_QUERY, NEIGHBOR_QUERY]
        if self.planner:
            plan = self.planner.plan(question)
            if plan:
                channels["cypher"] = self.repository.structured_search(plan)
                weights = {"vector": 0.45, "fulltext": 0.25, "cypher": 0.3}
                queries.extend([plan.cypher, SOURCE_QUERY])
        hits = reciprocal_rank_fusion(
            channels,
            weights=weights,
            limit=self.result_limit,
        )
        nodes, edges = self.repository.expand(
            [hit.chunk_id for hit in hits], max_hops=2
        )
        # Every graph fact provided to the generator must have its actual
        # textual evidence available as an F-numbered source.
        known = {hit.chunk_id for hit in hits}
        missing = [edge.chunk_id for edge in edges if edge.chunk_id not in known]
        if any(missing):
            hits.extend(self.repository.sources([item for item in missing if item]))
            queries.append(SOURCE_QUERY)
        known = {hit.chunk_id for hit in hits}
        edges = [edge for edge in edges if edge.chunk_id in known]
        return RetrievalResult(
            question=question,
            hits=hits,
            nodes=nodes,
            edges=edges,
            queries=queries,
            confidence=_retrieval_confidence(hits, len(edges)),
            warnings=list(warnings) if isinstance(warnings, list) else [],
        )


def _retrieval_confidence(hits, edge_count: int) -> float:
    if not hits:
        return 0.0
    top_score = hits[0].score
    channel_coverage = max(len(hit.channels) for hit in hits) / 2
    graph_signal = min(edge_count / 5, 1.0)
    confidence = 0.15 + 0.5 * top_score + 0.2 * channel_coverage + 0.15 * graph_signal
    return round(min(max(confidence, 0.0), 0.99), 3)
