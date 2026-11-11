from __future__ import annotations

import unittest

from trilhagraph.retrieval.algorithms import GraphAlgorithms, _calculate_pagerank
from trilhagraph.retrieval.fusion import reciprocal_rank_fusion
from trilhagraph.retrieval.generation import AnswerGenerator
from trilhagraph.retrieval.models import (
    GraphEdge,
    GraphNode,
    RetrievalResult,
    SearchHit,
)
from trilhagraph.retrieval.repository import RetrievalRepository, _fulltext_query
from trilhagraph.retrieval.retriever import HybridRetriever


def hit(chunk_id: str, score: float, channel: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        text=f"texto {chunk_id}",
        document_id=f"doc-{chunk_id}",
        title=f"Titulo {chunk_id}",
        filename=f"{chunk_id}.txt",
        score=score,
        channels={channel},
    )


class FusionTests(unittest.TestCase):
    def test_fuses_channels_and_rewards_shared_hit(self) -> None:
        result = reciprocal_rank_fusion(
            {
                "vector": [hit("shared", 0.9, "vector"), hit("v", 0.8, "vector")],
                "fulltext": [
                    hit("shared", 5.0, "fulltext"),
                    hit("f", 4.0, "fulltext"),
                ],
            },
            weights={"vector": 0.65, "fulltext": 0.35},
        )

        self.assertEqual(result[0].chunk_id, "shared")
        self.assertEqual(result[0].channels, {"vector", "fulltext"})
        self.assertLessEqual(result[0].score, 1.0)


class FakeResult:
    def __init__(self, rows):
        self.result_set = rows


class ExpansionGraph:
    def __init__(self) -> None:
        self.neighbor_calls = 0

    def ro_query(self, query, params):
        if "MENCIONA" in query:
            return FakeResult([["a", "Curso", "Curso A"]])
        self.neighbor_calls += 1
        if self.neighbor_calls == 1:
            return FakeResult(
                [
                    [
                        "a",
                        "Curso",
                        "Curso A",
                        "b",
                        "Area",
                        "Cidade B",
                        "a",
                        "b",
                        "ABORDA",
                        0.9,
                        "t1",
                    ]
                ]
            )
        return FakeResult(
            [
                [
                    "b",
                    "Area",
                    "Cidade B",
                    "c",
                    "Area",
                    "Estado C",
                    "b",
                    "c",
                    "PERTENCE_A",
                    0.8,
                    "t2",
                ]
            ]
        )


class RepositoryTests(unittest.TestCase):
    def test_sanitizes_fulltext_query(self) -> None:
        self.assertEqual(
            _fulltext_query('"trilhas" em (Salvador)?'), "trilhas | Salvador"
        )

    def test_expands_two_hops_and_preserves_direction(self) -> None:
        nodes, edges = RetrievalRepository(ExpansionGraph()).expand(["t1"])

        self.assertEqual({node.id for node in nodes}, {"a", "b", "c"})
        self.assertEqual(len(edges), 2)
        self.assertEqual(edges[0].source, "a")
        self.assertEqual(edges[0].target, "b")


class FakeEmbeddingClient:
    def embed(self, **kwargs):
        return [[0.1, 0.2]]


class FakeRepository:
    def vector_search(self, embedding, *, limit):
        return [hit("shared", 0.9, "vector")]

    def fulltext_search(self, question, *, limit):
        return [hit("shared", 5.0, "fulltext")]

    def expand(self, chunk_ids, *, max_hops):
        return (
            [GraphNode("a", "Curso", "A")],
            [GraphEdge("a", "b", "ABORDA", 1.0, "shared")],
        )


class RetrieverTests(unittest.TestCase):
    def test_runs_hybrid_search_and_returns_trace(self) -> None:
        result = HybridRetriever(
            repository=FakeRepository(),
            ollama=FakeEmbeddingClient(),
            embedding_model="fake",
            embedding_dimension=2,
        ).retrieve("Onde fazer trilha?")

        self.assertEqual(result.hits[0].channels, {"vector", "fulltext"})
        self.assertEqual(len(result.queries), 4)
        self.assertGreater(result.confidence, 0.5)


class FakeChatClient:
    def chat_json(self, **kwargs):
        return {
            "claims": [
                {
                    "text": "A resposta usa a fonte.",
                    "citations": ["F1"],
                    "graph_edges": [],
                }
            ],
            "abstained": False,
            "caveats": ["Confirme as condicoes locais."],
        }


class GenerationTests(unittest.TestCase):
    def test_validates_citations(self) -> None:
        retrieval = RetrievalResult(
            question="Pergunta",
            hits=[hit("1", 0.9, "vector")],
            nodes=[],
            edges=[],
            queries=[],
            confidence=0.8,
        )
        answer = AnswerGenerator(FakeChatClient(), model="fake").generate(retrieval)

        self.assertEqual(answer.citations, ["F1"])
        self.assertIn("[F1]", answer.answer)


class AlgorithmGraph:
    def ro_query(self, query, params):
        if "shortestPath" in query:
            return FakeResult([[["a", "b"], ["ABORDA"], 1]])
        return FakeResult([["a", "Curso A", "Curso", 0.75]])


class AlgorithmTests(unittest.TestCase):
    def test_shortest_path_and_pagerank(self) -> None:
        algorithms = GraphAlgorithms(AlgorithmGraph())

        path = algorithms.shortest_path("a", "b")
        ranking = algorithms.pagerank(limit=5)

        self.assertIsNotNone(path)
        self.assertEqual(path.length, 1)
        self.assertEqual(ranking[0].id, "a")
        self.assertEqual(ranking[0].score, 0.75)

    def test_python_pagerank_fallback_is_normalized(self) -> None:
        scores = _calculate_pagerank({"a": {"b"}, "b": {"c"}, "c": set()})

        self.assertAlmostEqual(sum(scores.values()), 1.0)
        self.assertGreater(scores["c"], scores["a"])


if __name__ == "__main__":
    unittest.main()
