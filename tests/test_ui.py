from __future__ import annotations

import unittest

from licenciagraph.retrieval.models import GraphEdge, GraphNode
from licenciagraph.ui.filters import ProcessFilters
from licenciagraph.ui.graph_view import build_graph_view_data


class FilterTests(unittest.TestCase):
    def test_augments_question_with_explicit_filters(self) -> None:
        filters = ProcessFilters(
            activity="avicultura industrial", modality="LI", porte="Médio"
        )

        result = filters.augment_question("Quais documentos preciso?")

        self.assertIn("Quais documentos preciso?", result)
        self.assertIn("atividade: avicultura industrial", result)
        self.assertIn("modalidade: LI", result)
        self.assertIn("porte: Médio", result)

    def test_keeps_question_unchanged_without_filters(self) -> None:
        self.assertEqual(
            ProcessFilters().augment_question("  Minha pergunta  "), "Minha pergunta"
        )


class GraphViewTests(unittest.TestCase):
    def test_layout_is_deterministic_and_ignores_dangling_edges(self) -> None:
        nodes = [
            GraphNode("b", "Modalidade", "Licença de Instalação"),
            GraphNode("a", "Atividade", "Avicultura industrial"),
        ]
        edges = [
            GraphEdge("a", "b", "ENQUADRADA_EM"),
            GraphEdge("a", "missing", "EXIGE_CHECKLIST"),
        ]

        first = build_graph_view_data(nodes, edges)
        second = build_graph_view_data(list(reversed(nodes)), edges)

        self.assertEqual(first.positions, second.positions)
        self.assertEqual(len(first.edges), 1)
        self.assertEqual(first.edges[0].relationship_type, "ENQUADRADA_EM")


if __name__ == "__main__":
    unittest.main()
