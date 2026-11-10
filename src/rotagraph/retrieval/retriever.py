"""Orquestracao da recuperacao hibrida e multi-hop."""

from __future__ import annotations

from rotagraph.ingestion.ollama import OllamaClient
from rotagraph.retrieval.fusion import reciprocal_rank_fusion
from rotagraph.retrieval.models import RetrievalResult
from rotagraph.retrieval.repository import (
    FULLTEXT_QUERY,
    NEIGHBOR_QUERY,
    SEED_QUERY,
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
    ) -> None:
        self.repository = repository
        self.ollama = ollama
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.candidate_limit = candidate_limit
        self.result_limit = result_limit

    def retrieve(self, question: str) -> RetrievalResult:
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
        fulltext_hits = self.repository.fulltext_search(
            question, limit=self.candidate_limit
        )
        hits = reciprocal_rank_fusion(
            {"vector": vector_hits, "fulltext": fulltext_hits},
            weights={"vector": 0.65, "fulltext": 0.35},
            limit=self.result_limit,
        )
        nodes, edges = self.repository.expand(
            [hit.chunk_id for hit in hits], max_hops=2
        )
        return RetrievalResult(
            question=question,
            hits=hits,
            nodes=nodes,
            edges=edges,
            queries=[VECTOR_QUERY, FULLTEXT_QUERY, SEED_QUERY, NEIGHBOR_QUERY],
            confidence=_retrieval_confidence(hits, len(edges)),
        )


def _retrieval_confidence(hits, edge_count: int) -> float:
    if not hits:
        return 0.0
    top_score = hits[0].score
    channel_coverage = max(len(hit.channels) for hit in hits) / 2
    graph_signal = min(edge_count / 5, 1.0)
    confidence = 0.15 + 0.5 * top_score + 0.2 * channel_coverage + 0.15 * graph_signal
    return round(min(max(confidence, 0.0), 0.99), 3)
