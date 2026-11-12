"""Regressões de rastreabilidade e consultas geradas."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import Mock

from licenciagraph.evaluation.evaluator import EvaluationCase, evaluate_answer
from licenciagraph.retrieval.generation import AnswerGenerator, _build_context, _schema
from licenciagraph.retrieval.models import (
    GraphEdge,
    GraphNode,
    GroundedAnswer,
    RetrievalResult,
    SearchHit,
)
from licenciagraph.retrieval.planning import CYPHER_CHOICES, CypherPlan, CypherPlanner
from licenciagraph.retrieval.repository import RetrievalRepository


def context():
    return RetrievalResult(
        question="Onde fica?",
        hits=[
            SearchHit("t1", "A fica em B.", "d1", "Fonte", "same.txt", 1.0),
            SearchHit("t2", "B faz parte de C.", "d1", "Fonte", "same.txt", 1.0),
        ],
        nodes=[GraphNode("a", "Curso", "A"), GraphNode("b", "Area", "B")],
        edges=[GraphEdge("a", "b", "ABORDA", 1.0, "t1")],
        queries=[],
        confidence=0.8,
    )


class GroundingRegressions(unittest.TestCase):
    def test_citation_ids_refer_to_chunks_not_unique_filenames(self):
        answer = GroundedAnswer("B faz parte de C [F2].", ["F2"], [], context())
        case = EvaluationCase("q", "Onde?", ("t1", "t2"), ("C",))
        self.assertEqual(evaluate_answer(case, answer).citation_validity, 1.0)

    def test_rejects_invalid_inline_and_unknown_citation(self):
        for text, citations in [
            ("Fato [F99].", ["F1"]),
            ("Fato.", ["F99"]),
        ]:
            client = Mock()
            client.chat_json.return_value = {
                "claims": [{"text": text, "citations": citations, "graph_edges": []}],
                "abstained": False,
                "caveats": [],
            }
            with (
                self.subTest(text=text, citations=citations),
                self.assertRaises(ValueError),
            ):
                AnswerGenerator(client, model="fake").generate(context())

    def test_empty_context_does_not_invoke_model(self):
        client = Mock()
        result = AnswerGenerator(client, model="fake").generate(
            replace(context(), hits=[])
        )
        self.assertFalse(result.citations)
        client.chat_json.assert_not_called()

    def test_pagerank_and_source_mapping_enter_generation_context(self):
        text = _build_context(context())
        self.assertIn("PAGERANK", text)
        self.assertIn("fonte=F1", text)
        self.assertIn("b: 0.", text)

    def test_arbitrary_model_query_is_rejected_before_execution(self):
        client, graph = Mock(), Mock()
        client.chat_json.return_value = {
            "cypher": "MATCH (n) DETACH DELETE n",
            "name": "A",
        }
        self.assertIsNone(CypherPlanner(client, model="fake").plan("A"))
        with self.assertRaises(ValueError):
            RetrievalRepository(graph).structured_search(
                CypherPlan("CALL db.labels()", "A")
            )
        graph.ro_query.assert_not_called()

    def test_vector_schema_has_no_graph_edges_field(self):
        schema = _schema(replace(context(), edges=[]))
        claim = schema["properties"]["claims"]["items"]
        self.assertNotIn("graph_edges", claim["properties"])
        self.assertEqual(claim["required"], ["text", "citations"])

    def test_abstention_without_claims_is_accepted(self):
        client = Mock()
        client.chat_json.return_value = {
            "claims": [],
            "abstained": True,
            "caveats": [],
        }
        result = AnswerGenerator(client, model="fake").generate(context())
        self.assertTrue(result.abstained)
        self.assertFalse(result.citations)
        self.assertTrue(client.chat_json.call_args.kwargs["strict"])

    def test_generated_name_is_passed_as_parameter(self):
        graph = Mock()
        graph.ro_query.return_value.result_set = []
        injected = "' DELETE n //"
        RetrievalRepository(graph).structured_search(
            CypherPlan(CYPHER_CHOICES[0], injected)
        )
        call = graph.ro_query.call_args
        self.assertNotIn(injected, call.args[0])
        self.assertEqual(call.kwargs["params"], {"name": injected})


class HybridPlanRegression(unittest.TestCase):
    def test_cypher_and_graph_evidence_reach_generator(self):
        from licenciagraph.retrieval.retriever import HybridRetriever

        repository, planner, embedding = Mock(), Mock(), Mock()
        plan = CypherPlan(CYPHER_CHOICES[0], "Salvador")
        planner.plan.return_value = plan
        embedding.embed.return_value = [[0.1, 0.2]]
        base = context().hits[0]
        graph_source = replace(base, chunk_id="t3", text="Fato adicional.")
        repository.vector_search.return_value = [base]
        repository.fulltext_search.return_value = []
        repository.structured_search.return_value = [base]
        repository.expand.return_value = (
            context().nodes,
            [GraphEdge("a", "b", "ABORDA", 1.0, "t3")],
        )
        repository.sources.return_value = [graph_source]
        repository.linked_entities.return_value = []
        repository.similar_activities.return_value = []
        repository.activity_communities.return_value = []
        result = HybridRetriever(
            repository=repository,
            llm=embedding,
            embedding_model="fake",
            embedding_dimension=2,
            planner=planner,
        ).retrieve("Onde fica Salvador?")
        repository.structured_search.assert_called_once_with(plan)
        repository.sources.assert_called_once_with(["t3"])
        self.assertEqual({hit.chunk_id for hit in result.hits}, {"t1", "t3"})
        self.assertIn(plan.cypher, result.queries)
        self.assertIn("fonte=F2", _build_context(result))


class BaselineIsolationTests(unittest.TestCase):
    def test_vector_baseline_does_not_consult_graph_or_planner(self):
        from licenciagraph.retrieval.retriever import HybridRetriever

        repository, planner, embedding = Mock(), Mock(), Mock()
        embedding.embed.return_value = [[0.1, 0.2]]
        repository.vector_search.return_value = context().hits
        answer = HybridRetriever(
            repository=repository,
            llm=embedding,
            planner=planner,
            embedding_model="fake",
            embedding_dimension=2,
            mode="vector",
        ).retrieve("Onde fica?")
        self.assertFalse(answer.edges)
        self.assertEqual(len(answer.hits), 2)
        repository.fulltext_search.assert_not_called()
        repository.expand.assert_not_called()
        planner.plan.assert_not_called()


class AlgorithmCompatibilityTests(unittest.TestCase):
    def test_bfs_fallback_respects_hop_limit_and_reverse_edges(self):
        from licenciagraph.retrieval.algorithms import GraphAlgorithms

        graph = Mock()

        def query(text, params=None):
            if "shortestPath" in text:
                raise RuntimeError(
                    "FalkorDB does not currently support "
                    "undirected shortestPath traversals"
                )
            return Mock(
                result_set=[
                    ["a", "b", "PRE_REQUISITO"],
                    ["c", "b", "ABORDA"],
                    ["c", "d", "DESENVOLVE"],
                ]
            )

        graph.ro_query.side_effect = query
        algorithms = GraphAlgorithms(graph)
        self.assertIsNone(algorithms.shortest_path("a", "d", max_hops=2))
        path = algorithms.shortest_path("a", "d", max_hops=3)
        self.assertEqual(path.node_ids, ["a", "b", "c", "d"])
        self.assertEqual(path.length, 3)


class FactsExtractionTests(unittest.TestCase):
    def test_accepts_grounded_facts_with_known_modalities(self):
        from licenciagraph.ingestion.facts import parse_facts

        text = (
            "A Licença Prévia é emitida pela SUDEMA e regulada pela Lei nº 6.938/1981."
        )
        result = parse_facts(
            {
                "facts": [
                    {"relation": "EMITIDA_POR", "subject": "LP", "object": "SUDEMA"},
                    {
                        "relation": "REGULADO_POR",
                        "subject": "LP",
                        "object": "Lei nº 6.938/1981",
                    },
                ]
            },
            text,
        )
        labels = {entity.ref: entity.label for entity in result.entities}
        pairs = {
            (labels[r.source_ref], r.relationship_type, labels[r.target_ref])
            for r in result.relationships
        }
        self.assertEqual(
            pairs,
            {
                ("Modalidade", "EMITIDA_POR", "Orgao"),
                ("Modalidade", "REGULADO_POR", "Norma"),
            },
        )

    def test_rejects_entities_absent_from_the_text(self):
        from licenciagraph.ingestion.facts import parse_facts

        with self.assertRaisesRegex(ValueError, "ausente no texto"):
            parse_facts(
                {
                    "facts": [
                        {"relation": "EMITIDA_POR", "subject": "LP", "object": "IBAMA"}
                    ]
                },
                "A Licença Prévia é emitida pela SUDEMA.",
            )

    def test_rejects_unknown_modality(self):
        from licenciagraph.ingestion.facts import parse_facts

        with self.assertRaisesRegex(ValueError, "Modalidade desconhecida"):
            parse_facts(
                {
                    "facts": [
                        {
                            "relation": "EMITIDA_POR",
                            "subject": "XYZ",
                            "object": "SUDEMA",
                        }
                    ]
                },
                "XYZ SUDEMA",
            )

    def test_retries_invalid_extraction_with_feedback(self):
        from licenciagraph.ingestion.facts import FactsExtractor

        client = Mock()
        invalid = {
            "facts": [{"relation": "EMITIDA_POR", "subject": "LP", "object": "IBAMA"}]
        }
        valid = {
            "facts": [{"relation": "EMITIDA_POR", "subject": "LP", "object": "SUDEMA"}]
        }
        client.chat_json.side_effect = [invalid, valid]
        result = FactsExtractor(client, model="fake").extract(
            "A Licença Prévia é emitida pela SUDEMA."
        )
        self.assertEqual(len(result.relationships), 1)
        self.assertEqual(client.chat_json.call_count, 2)
        self.assertIn("ausente no texto", client.chat_json.call_args.kwargs["prompt"])


class ComparisonMetricTests(unittest.TestCase):
    def test_content_checks_do_not_require_graph_and_extra_sources_cost_precision(self):
        retrieval = replace(
            context(),
            edges=[],
            hits=[
                context().hits[0],
                replace(context().hits[1], chunk_id="irrelevante"),
            ],
        )
        answer = GroundedAnswer("Python [F1].", ["F1"], [], retrieval)
        case = EvaluationCase("q", "Pergunta", ("t1",), ("Python",), True)
        result = evaluate_answer(case, answer)
        self.assertTrue(result.content_checks_passed)
        self.assertFalse(result.passed)
        self.assertEqual(result.source_precision, 0.5)


class StructuredAnswerTests(unittest.TestCase):
    def test_claims_with_reais_not_in_the_cited_source_are_dropped(self):
        client = Mock()
        client.chat_json.return_value = {
            "claims": [
                {"text": "A análise custa R$ 1.200,00.", "citations": ["F1"]},
                {"text": "A letra de custo é a F.", "citations": ["F1"]},
            ],
            "abstained": False,
            "caveats": [],
        }
        answer = AnswerGenerator(client, model="fake").generate(
            replace(context(), question="Quanto custa a análise em reais?")
        )
        self.assertNotIn("1.200", answer.answer)
        self.assertIn("letra de custo", answer.answer)

    def test_ufr_notice_is_added_to_cost_answers(self):
        client = Mock()
        client.chat_json.return_value = {
            "claims": [{"text": "A análise custa 16 UFR-PB.", "citations": ["F1"]}],
            "abstained": False,
            "caveats": [],
        }
        answer = AnswerGenerator(client, model="fake").generate(
            replace(context(), question="Quanto custa a análise da LAU?")
        )
        self.assertIn("16 UFR-PB", answer.answer)
        self.assertIn("conversão para reais", answer.answer)

    def test_ufr_notice_is_not_added_to_other_questions(self):
        client = Mock()
        client.chat_json.return_value = {
            "claims": [{"text": "A validade é de 5 anos.", "citations": ["F1"]}],
            "abstained": False,
            "caveats": [],
        }
        answer = AnswerGenerator(client, model="fake").generate(context())
        self.assertNotIn("conversão para reais", answer.answer)

    def test_formats_declared_sources_and_retains_selected_edges(self):
        client = Mock()
        client.chat_json.return_value = {
            "claims": [
                {"text": "Fato sustentado.", "citations": ["F1"], "graph_edges": ["G1"]}
            ],
            "abstained": False,
            "caveats": [],
        }
        answer = AnswerGenerator(client, model="fake").generate(context())
        self.assertIn("[F1]", answer.answer)
        self.assertEqual(answer.supporting_edges, ["G1"])

    def test_explicit_abstention_can_have_no_sources(self):
        client = Mock()
        client.chat_json.return_value = {
            "claims": [
                {
                    "text": "O catálogo não informa isso.",
                    "citations": [],
                    "graph_edges": [],
                }
            ],
            "abstained": True,
            "caveats": [],
        }
        answer = AnswerGenerator(client, model="fake").generate(context())
        self.assertTrue(answer.abstained)
        self.assertFalse(answer.citations)


class CaveatGroundingTests(unittest.TestCase):
    def test_ungrounded_model_caveat_is_not_presented_as_a_fact(self):
        client = Mock()
        client.chat_json.return_value = {
            "claims": [
                {"text": "Fato sustentado.", "citations": ["F1"], "graph_edges": []}
            ],
            "abstained": False,
            "caveats": ["O curso exige a si próprio."],
        }
        answer = AnswerGenerator(client, model="fake").generate(context())
        self.assertNotIn("O curso exige a si próprio.", answer.caveats)
        self.assertIn("orientativa", answer.caveats[0])
        self.assertIn("Fato sustentado.", answer.answer)


class QueryTimeoutFallbackTests(unittest.TestCase):
    def test_optional_query_timeout_preserves_other_channels_and_error_visibility(self):
        graph = Mock()
        graph.ro_query.side_effect = RuntimeError("Query timed out")
        repository = RetrievalRepository(graph)
        result = repository.structured_search(CypherPlan(CYPHER_CHOICES[0], "Python"))
        self.assertEqual(result, [])
        self.assertIn("excedeu o prazo", repository.warnings[0])
        graph.ro_query.side_effect = RuntimeError("connection lost")
        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            repository.structured_search(CypherPlan(CYPHER_CHOICES[0], "Python"))


class VerificationTests(unittest.TestCase):
    def _claims(self, *texts):
        return {
            "claims": [{"text": text, "citations": ["F1"]} for text in texts],
            "abstained": False,
            "caveats": [],
        }

    def test_unsupported_claims_are_dropped(self):
        client = Mock()
        client.chat_json.side_effect = [
            self._claims("Fato real.", "Fato inventado."),
            {
                "verdicts": [
                    {"index": 0, "supported": True},
                    {"index": 1, "supported": False},
                ],
                "missing": [],
            },
        ]
        answer = AnswerGenerator(client, model="m", verifier_model="v").generate(
            context()
        )
        self.assertIn("Fato real", answer.answer)
        self.assertNotIn("inventado", answer.answer)
        self.assertEqual(client.chat_json.call_args.kwargs["model"], "v")

    def test_missing_aspects_trigger_one_revision(self):
        client = Mock()
        client.chat_json.side_effect = [
            self._claims("Parcial."),
            {
                "verdicts": [{"index": 0, "supported": True}],
                "missing": ["competências"],
            },
            self._claims("Parcial.", "Competência incluída."),
            {
                "verdicts": [
                    {"index": 0, "supported": True},
                    {"index": 1, "supported": True},
                ],
                "missing": ["algo ainda faltando"],
            },
        ]
        answer = AnswerGenerator(client, model="m", verifier_model="v").generate(
            context()
        )
        self.assertEqual(client.chat_json.call_count, 4)
        self.assertIn("Competência incluída", answer.answer)
        self.assertIn(
            "competências", client.chat_json.call_args_list[2].kwargs["prompt"]
        )

    def test_all_claims_unsupported_becomes_abstention(self):
        client = Mock()
        client.chat_json.side_effect = [
            self._claims("Inventado."),
            {"verdicts": [{"index": 0, "supported": False}], "missing": []},
        ]
        answer = AnswerGenerator(client, model="m", verifier_model="v").generate(
            context()
        )
        self.assertTrue(answer.abstained)
        self.assertFalse(answer.citations)

    def test_claim_without_citation_is_dropped_not_fatal(self):
        client = Mock()
        client.chat_json.return_value = {
            "claims": [
                {"text": "Sem fonte.", "citations": []},
                {"text": "Com fonte.", "citations": ["F1"]},
            ],
            "abstained": False,
            "caveats": [],
        }
        answer = AnswerGenerator(client, model="m").generate(context())
        self.assertNotIn("Sem fonte", answer.answer)
        self.assertIn("Com fonte", answer.answer)

    def test_paths_and_communities_enter_the_context(self):
        text = _build_context(
            replace(
                context(),
                paths=[["A", "B", "C"]],
                communities=[["A", "B"]],
            )
        )
        self.assertIn(
            "SEQUÊNCIA DE LICENÇAS (caminho mais curto por PRECEDE): A -> B -> C", text
        )
        self.assertIn("COMUNIDADE DE LOUVAIN", text.upper())
