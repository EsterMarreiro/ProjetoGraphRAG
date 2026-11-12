"""Contratos da recuperacao e da resposta fundamentada."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    text: str
    document_id: str
    title: str
    filename: str
    score: float
    channels: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    label: str
    name: str


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    target: str
    relationship_type: str
    confidence: float | None = None
    chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    question: str
    hits: list[SearchHit]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    queries: list[str]
    confidence: float
    warnings: list[str] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)
    communities: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    answer: str
    citations: list[str]
    caveats: list[str]
    retrieval: RetrievalResult
    supporting_edges: list[str] = field(default_factory=list)
    abstained: bool = False
