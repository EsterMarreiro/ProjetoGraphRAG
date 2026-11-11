"""Algoritmos de grafo expostos para recuperacao e explicabilidade."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PathResult:
    node_ids: list[str]
    relationship_types: list[str]
    length: int


@dataclass(frozen=True, slots=True)
class RankedNode:
    id: str
    name: str
    label: str
    score: float


class GraphAlgorithms:
    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def shortest_path(
        self, source_id: str, target_id: str, *, max_hops: int = 6
    ) -> PathResult | None:
        if not 1 <= max_hops <= 10:
            raise ValueError("max_hops deve estar entre 1 e 10")
        query = f"""
            MATCH (source {{id: $source_id}}), (target {{id: $target_id}})
            WITH shortestPath((source)-[*..{max_hops}]-(target)) AS path
            RETURN [node IN nodes(path) | node.id],
                   [relationship IN relationships(path) | type(relationship)],
                   length(path)
        """
        try:
            rows = self.graph.ro_query(
                query, params={"source_id": source_id, "target_id": target_id}
            ).result_set
        except Exception as exc:
            if (
                "does not currently support undirected shortestpath"
                not in str(exc).lower()
            ):
                raise
            return self._shortest_path_fallback(source_id, target_id, max_hops)
        if not rows or rows[0][0] is None:
            return None
        return PathResult(
            node_ids=[str(item) for item in rows[0][0]],
            relationship_types=[str(item) for item in rows[0][1]],
            length=int(rows[0][2]),
        )

    def _shortest_path_fallback(self, source: str, target: str, max_hops: int):
        """BFS on domain edges for engines lacking undirected shortestPath."""
        query = """
            MATCH (a)-[r]->(b)
            WHERE NOT a:Trecho AND NOT a:Documento
              AND NOT b:Trecho AND NOT b:Documento
            RETURN a.id, b.id, type(r)
        """
        adjacency: dict[str, set[tuple[str, str]]] = {}
        for a, b, relation in self.graph.ro_query(query).result_set:
            adjacency.setdefault(a, set()).add((b, relation))
            adjacency.setdefault(b, set()).add((a, relation))
        if source not in adjacency or target not in adjacency:
            return None
        frontier = deque([(source, [source], [])])
        visited = {source}
        while frontier:
            current, nodes, relationships = frontier.popleft()
            if current == target:
                return PathResult(nodes, relationships, len(relationships))
            if len(relationships) >= max_hops:
                continue
            for neighbor, relation in sorted(adjacency[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(
                        (neighbor, [*nodes, neighbor], [*relationships, relation])
                    )
        return None

    def pagerank(self, *, limit: int = 10) -> list[RankedNode]:
        if not 1 <= limit <= 100:
            raise ValueError("limit deve estar entre 1 e 100")
        query = """
            CALL pagerank.stream() YIELD node, score
            WHERE NOT node:Trecho AND NOT node:Documento
            RETURN node.id, coalesce(node.nome, node.id), labels(node)[0], score
            ORDER BY score DESC
            LIMIT $limit
        """
        try:
            rows = self.graph.ro_query(query, params={"limit": limit}).result_set
            return [
                RankedNode(str(row[0]), str(row[1]), str(row[2]), float(row[3]))
                for row in rows
            ]
        except Exception as exc:
            message = str(exc).casefold()
            unsupported = (
                "unknown function 'pagerank.stream'",
                "procedure `pagerank.stream` is not registered",
            )
            if not any(marker in message for marker in unsupported):
                raise
            return self._pagerank_fallback(limit=limit)

    def _pagerank_fallback(self, *, limit: int) -> list[RankedNode]:
        query = """
            MATCH (source)-[relationship]->(target)
            WHERE NOT source:Trecho AND NOT source:Documento
              AND NOT target:Trecho AND NOT target:Documento
            RETURN source.id, coalesce(source.nome, source.id), labels(source)[0],
                   target.id, coalesce(target.nome, target.id), labels(target)[0]
        """
        rows = self.graph.ro_query(query).result_set
        metadata: dict[str, tuple[str, str]] = {}
        outgoing: dict[str, set[str]] = {}
        for row in rows:
            source_id, target_id = str(row[0]), str(row[3])
            metadata[source_id] = (str(row[1]), str(row[2]))
            metadata[target_id] = (str(row[4]), str(row[5]))
            outgoing.setdefault(source_id, set()).add(target_id)
            outgoing.setdefault(target_id, set())
        scores = _calculate_pagerank(outgoing)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [
            RankedNode(node_id, metadata[node_id][0], metadata[node_id][1], score)
            for node_id, score in ordered
        ]


def _calculate_pagerank(
    outgoing: dict[str, set[str]],
    *,
    damping: float = 0.85,
    iterations: int = 50,
) -> dict[str, float]:
    if not outgoing:
        return {}
    node_ids = sorted(outgoing)
    count = len(node_ids)
    scores = {node_id: 1.0 / count for node_id in node_ids}
    for _ in range(iterations):
        dangling = sum(scores[node_id] for node_id in node_ids if not outgoing[node_id])
        updated = {
            node_id: (1.0 - damping) / count + damping * dangling / count
            for node_id in node_ids
        }
        for source_id, targets in outgoing.items():
            if not targets:
                continue
            contribution = damping * scores[source_id] / len(targets)
            for target_id in targets:
                updated[target_id] += contribution
        scores = updated
    return scores
