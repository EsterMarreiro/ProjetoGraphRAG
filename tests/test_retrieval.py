from __future__ import annotations

import unittest
import unittest.mock

from licenciagraph.retrieval.algorithms import GraphAlgorithms, _calculate_pagerank
from licenciagraph.retrieval.fusion import reciprocal_rank_fusion
from licenciagraph.retrieval.generation import AnswerGenerator
from licenciagraph.retrieval.models import (
    GraphEdge,
    GraphNode,
    RetrievalResult,
    SearchHit,
)
from licenciagraph.retrieval.repository import RetrievalRepository, _fulltext_query
from licenciagraph.retrieval.retriever import HybridRetriever


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


def edge_row(source, source_label, target, target_label, relation, chunk, props=None):
    return [
        source,
        source_label,
        f"Nome {source}",
        target,
        target_label,
        f"Nome {target}",
        source,
        target,
        relation,
        1.0,
        chunk,
        props or {},
    ]


class ExpansionGraph:
    """Devolve vizinhos por rótulo: atividade, depois checklist, depois documento."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def ro_query(self, query, params):
        self.queries.append(query)
        if "MENCIONA" in query:
            return FakeResult([["a", "Atividade", "Atividade A"]])
        if "(focus:Atividade" in query:
            return FakeResult(
                [edge_row("a", "Atividade", "c", "Checklist", "EXIGE_CHECKLIST", "t1")]
            )
        if "(focus:Checklist" in query:
            return FakeResult(
                [
                    edge_row(
                        "c",
                        "Checklist",
                        "d",
                        "DocumentoExigido",
                        "EXIGE_DOCUMENTO",
                        "t2",
                        {"ordem": 4},
                    )
                ]
            )
        return FakeResult([])


class RepositoryTests(unittest.TestCase):
    def test_fulltext_query_drops_question_words(self) -> None:
        self.assertEqual(
            _fulltext_query("Onde fica a SUDEMA em Campina Grande?"),
            "SUDEMA | Campina | Grande",
        )

    def test_sanitizes_fulltext_query(self) -> None:
        self.assertEqual(
            _fulltext_query('"licença" de (Instalação)?'), "licença | Instalação"
        )

    def test_expansion_follows_outgoing_edges_and_keeps_properties(self) -> None:
        graph = ExpansionGraph()
        repository = RetrievalRepository(graph)
        nodes, edges = repository.expand(
            ["t1"],
            seeds=[
                GraphNode("a", "Atividade", "Atividade A"),
                GraphNode(
                    "modalidade:li",
                    "Modalidade",
                    "Licença de Instalação (LI)",
                    ("Licença de Instalação", "LI"),
                ),
            ],
            max_hops=3,
        )

        self.assertEqual({node.id for node in nodes}, {"a", "c", "d", "modalidade:li"})
        self.assertEqual(
            [edge.relationship_type for edge in edges][:1], ["EXIGE_CHECKLIST"]
        )
        self.assertEqual(edges[1].properties, (("ordem", "4"),))

    def test_checklists_are_not_expanded_without_a_modality(self) -> None:
        graph = ExpansionGraph()
        nodes, _ = RetrievalRepository(graph).expand(
            ["t1"], seeds=[GraphNode("a", "Atividade", "Atividade A")]
        )

        self.assertNotIn("d", {node.id for node in nodes})

    def test_label_is_validated_before_building_a_query(self) -> None:
        from licenciagraph.retrieval.repository import OUTGOING_QUERY, query_for_label

        with self.assertRaises(ValueError):
            query_for_label(OUTGOING_QUERY, "A) DETACH DELETE n //")


class FakeEmbeddingClient:
    def embed(self, **kwargs):
        return [[0.1, 0.2]]


class FakeRepository:
    def vector_search(self, embedding, *, limit):
        return [hit("shared", 0.9, "vector")]

    def fulltext_search(self, question, *, limit):
        return [hit("shared", 5.0, "fulltext")]

    def linked_entities(self, question):
        return [GraphNode("modalidade:li", "Modalidade", "Licença de Instalação")]

    def similar_activities(self, embedding):
        return []

    def precedence_path(self, source, target):
        return []

    def activity_communities(self):
        return []

    def expand(self, chunk_ids, *, max_hops, seeds=None, seed_ids=None):
        return (
            [GraphNode("a", "Atividade", "A")],
            [GraphEdge("a", "b", "ENQUADRADA_EM", 1.0, "shared")],
        )


class RetrieverTests(unittest.TestCase):
    def test_runs_hybrid_search_and_returns_trace(self) -> None:
        result = HybridRetriever(
            repository=FakeRepository(),
            llm=FakeEmbeddingClient(),
            embedding_model="fake",
            embedding_dimension=2,
        ).retrieve("Que documentos a LI exige?")

        self.assertEqual(result.hits[0].channels, {"vector", "fulltext"})
        self.assertEqual(len(result.queries), 4)
        self.assertEqual(result.edges[0].relationship_type, "ENQUADRADA_EM")
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


class EntityLinkingAndAlgorithmsTests(unittest.TestCase):
    def _repository(self, rows):
        graph = unittest.mock.Mock()
        graph.ro_query.return_value.result_set = [
            [f"id:{i}", label, name, sigla, code]
            for i, (label, name, sigla, code) in enumerate(rows)
        ]
        return RetrievalRepository(graph)

    def test_linking_ignores_names_contained_in_longer_names(self):
        repository = self._repository(
            [
                ("Subgrupo", "AVICULTURA", None, None),
                ("Atividade", "Avicultura industrial (postura e corte)", None, None),
                ("Orgao", "SUDEMA", "SUDEMA", None),
            ]
        )
        linked = repository.linked_entities(
            "A avicultura industrial precisa de licença da SUDEMA?"
        )
        self.assertEqual(
            {node.name for node in linked},
            {"Avicultura industrial (postura e corte)", "SUDEMA"},
        )

    def test_linking_is_accent_and_case_insensitive(self):
        repository = self._repository(
            [("Estudo", "Relatório de Controle Ambiental", None, None)]
        )
        self.assertEqual(
            [
                n.name
                for n in repository.linked_entities(
                    "precisa de RELATORIO DE CONTROLE AMBIENTAL?"
                )
            ],
            ["Relatório de Controle Ambiental"],
        )

    def test_modality_is_linked_by_name_and_by_uppercase_acronym(self):
        repository = self._repository(
            [("Modalidade", "Licença de Instalação (LI)", "LI", None)]
        )
        by_name = repository.linked_entities(
            "Qual a validade da licença de instalação?"
        )
        by_acronym = repository.linked_entities("Documentos da LI para obras")
        lowercase = repository.linked_entities("eu li o manual")
        self.assertEqual([n.id for n in by_name], ["id:0"])
        self.assertEqual([n.id for n in by_acronym], ["id:0"])
        self.assertEqual(lowercase, [])

    def test_checklist_is_linked_by_code(self):
        repository = self._repository(
            [("Checklist", "AG-02 · Licença de Instalação (LI)", None, "AG-02")]
        )
        self.assertEqual(len(repository.linked_entities("itens do checklist AG-02")), 1)

    def test_louvain_groups_connected_activities_deterministically(self):
        from licenciagraph.retrieval.algorithms import louvain_communities

        edges = [("a", "b"), ("b", "c"), ("a", "c"), ("x", "y")]
        groups = louvain_communities(edges)
        self.assertEqual(groups, louvain_communities(edges))
        self.assertIn(["a", "b", "c"], groups)
        self.assertIn(["x", "y"], groups)

    def test_typed_shortest_path_rejects_unsafe_relationship(self):
        from licenciagraph.retrieval.algorithms import GraphAlgorithms

        with self.assertRaises(ValueError):
            GraphAlgorithms(unittest.mock.Mock()).shortest_path(
                "a", "b", relationship_type="X] DETACH DELETE n //"
            )
