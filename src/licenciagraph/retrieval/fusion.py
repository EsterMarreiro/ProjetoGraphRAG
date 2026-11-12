"""Fusao transparente dos rankings vetorial e textual."""

from __future__ import annotations

from licenciagraph.retrieval.models import SearchHit


def reciprocal_rank_fusion(
    rankings: dict[str, list[SearchHit]],
    *,
    weights: dict[str, float] | None = None,
    rank_constant: int = 60,
    limit: int = 8,
) -> list[SearchHit]:
    if rank_constant <= 0:
        raise ValueError("rank_constant deve ser maior que zero")
    if limit <= 0:
        raise ValueError("limit deve ser maior que zero")
    weights = weights or {channel: 1.0 for channel in rankings}

    combined: dict[str, SearchHit] = {}
    rrf_scores: dict[str, float] = {}
    for channel, hits in rankings.items():
        weight = weights.get(channel, 1.0)
        if weight < 0:
            raise ValueError("Pesos de ranking nao podem ser negativos")
        for rank, hit in enumerate(hits, start=1):
            if hit.chunk_id not in combined:
                combined[hit.chunk_id] = SearchHit(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    document_id=hit.document_id,
                    title=hit.title,
                    filename=hit.filename,
                    score=0.0,
                    channels=set(),
                    page=hit.page,
                )
            combined[hit.chunk_id].channels.add(channel)
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (
                weight / (rank_constant + rank)
            )

    maximum = sum(
        weight / (rank_constant + 1) for weight in weights.values() if weight > 0
    )
    for chunk_id, hit in combined.items():
        hit.score = rrf_scores[chunk_id] / maximum if maximum else 0.0

    return sorted(combined.values(), key=lambda hit: (-hit.score, hit.chunk_id))[:limit]
