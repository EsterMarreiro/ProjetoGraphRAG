from __future__ import annotations

import unittest

from rotagraph.retrieval.models import GraphEdge, GraphNode
from rotagraph.ui.filters import TourismFilters
from rotagraph.ui.graph_view import build_graph_view_data


class FilterTests(unittest.TestCase):
    def test_augments_question_with_explicit_filters(self) -> None:
        filters = TourismFilters(
            regions=("Nordeste",),
            state="BA",
            activity="Gastronomia",
            profile="Famílias",
            period="Período seco",
        )

        result = filters.augment_question("Qual destino escolher?")

        self.assertIn("Qual destino escolher?", result)
        self.assertIn("regiões: Nordeste", result)
        self.assertIn("UF ou estado: BA", result)
        self.assertIn("atividade: Gastronomia", result)

    def test_keeps_question_unchanged_without_filters(self) -> None:
        self.assertEqual(
            TourismFilters().augment_question("  Minha pergunta  "), "Minha pergunta"
        )


class GraphViewTests(unittest.TestCase):
    def test_layout_is_deterministic_and_ignores_dangling_edges(self) -> None:
        nodes = [
            GraphNode("b", "Localidade", "Bahia"),
            GraphNode("a", "Atracao", "Pelourinho"),
        ]
        edges = [
            GraphEdge("a", "b", "LOCALIZADA_EM"),
            GraphEdge("a", "missing", "PROXIMA_DE"),
        ]

        first = build_graph_view_data(nodes, edges)
        second = build_graph_view_data(list(reversed(nodes)), edges)

        self.assertEqual(first.positions, second.positions)
        self.assertEqual(len(first.edges), 1)
        self.assertEqual(first.edges[0].relationship_type, "LOCALIZADA_EM")


if __name__ == "__main__":
    unittest.main()
