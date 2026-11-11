from __future__ import annotations

import unittest

from trilhagraph.retrieval.models import GraphEdge, GraphNode
from trilhagraph.ui.filters import LearningFilters
from trilhagraph.ui.graph_view import build_graph_view_data


class FilterTests(unittest.TestCase):
    def test_augments_question_with_explicit_filters(self) -> None:
        filters = LearningFilters(
            area="Dados",
            competency="Recuperação vetorial",
            profile="Analistas de dados",
            level="Intermediário",
        )

        result = filters.augment_question("Qual curso escolher?")

        self.assertIn("Qual curso escolher?", result)
        self.assertIn("área: Dados", result)
        self.assertIn("nível: Intermediário", result)
        self.assertIn("competência: Recuperação vetorial", result)

    def test_keeps_question_unchanged_without_filters(self) -> None:
        self.assertEqual(
            LearningFilters().augment_question("  Minha pergunta  "), "Minha pergunta"
        )


class GraphViewTests(unittest.TestCase):
    def test_layout_is_deterministic_and_ignores_dangling_edges(self) -> None:
        nodes = [
            GraphNode("b", "Area", "Dados"),
            GraphNode("a", "Curso", "Grafos de Conhecimento"),
        ]
        edges = [
            GraphEdge("a", "b", "ABORDA"),
            GraphEdge("a", "missing", "PRE_REQUISITO"),
        ]

        first = build_graph_view_data(nodes, edges)
        second = build_graph_view_data(list(reversed(nodes)), edges)

        self.assertEqual(first.positions, second.positions)
        self.assertEqual(len(first.edges), 1)
        self.assertEqual(first.edges[0].relationship_type, "ABORDA")


if __name__ == "__main__":
    unittest.main()
