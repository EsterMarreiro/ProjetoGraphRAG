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
    page: int | None = None


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    label: str
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    target: str
    relationship_type: str
    confidence: float | None = None
    chunk_id: str | None = None
    properties: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Resolution:
    """Atividade descrita pelo empreendedor, conferida contra as atividades do grafo."""

    match: str
    activities: tuple[GraphNode, ...] = ()
    reason: str = ""
    missing: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    standalone: str = ""


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
    similar: list[str] = field(default_factory=list)
    linked_activities: list[str] = field(default_factory=list)
    resolution: Resolution | None = None


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    answer: str
    citations: list[str]
    caveats: list[str]
    retrieval: RetrievalResult
    supporting_edges: list[str] = field(default_factory=list)
    abstained: bool = False
