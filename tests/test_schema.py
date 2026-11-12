from __future__ import annotations

import unittest

from licenciagraph.graph.schema import ENTITY_LABELS, SchemaManager


class FakeGraph:
    def __init__(self, *, duplicate_indices: bool = False) -> None:
        self.duplicate_indices = duplicate_indices
        self.index_calls: list[tuple] = []
        self.constraints: list[dict[str, object]] = []

    def _index(self, kind: str, label: str, *properties: str, **options: object):
        self.index_calls.append((kind, label, properties, options))
        if self.duplicate_indices:
            raise RuntimeError("Index already exists")

    def create_node_range_index(self, label: str, *properties: str):
        return self._index("range", label, *properties)

    def create_node_fulltext_index(self, label: str, *properties: str):
        return self._index("fulltext", label, *properties)

    def create_node_vector_index(self, label: str, *properties: str, **options: object):
        return self._index("vector", label, *properties, **options)

    def list_constraints(self) -> list[dict[str, object]]:
        return list(self.constraints)

    def _constraint(self, kind: str, label: str, properties: tuple[str, ...]):
        self.constraints.append(
            {
                "type": kind,
                "label": label,
                "properties": list(properties),
                "entitytype": "NODE",
                "status": "UNDER CONSTRUCTION",
            }
        )

    def create_node_unique_constraint(self, label: str, *properties: str):
        self._constraint("UNIQUE", label, properties)

    def create_node_mandatory_constraint(self, label: str, *properties: str):
        self._constraint("MANDATORY", label, properties)


class SchemaManagerTests(unittest.TestCase):
    def test_creates_complete_schema(self) -> None:
        graph = FakeGraph()
        report = SchemaManager(graph, embedding_dimension=768).ensure()

        self.assertEqual(report.indices_created, 21)
        self.assertEqual(report.constraints_created, 25)
        unique_labels = {
            item["label"] for item in graph.constraints if item["type"] == "UNIQUE"
        }
        self.assertEqual(unique_labels, set(ENTITY_LABELS))
        vector_calls = [call for call in graph.index_calls if call[0] == "vector"]
        self.assertEqual(len(vector_calls), 3)
        self.assertTrue(
            all(call[3]["similarity_function"] == "cosine" for call in vector_calls)
        )

    def test_second_run_skips_existing_constraints_and_indices(self) -> None:
        graph = FakeGraph(duplicate_indices=True)
        manager = SchemaManager(graph)
        first = manager.ensure()
        second = manager.ensure()

        self.assertEqual(first.indices_existing, 21)
        self.assertEqual(first.constraints_created, 25)
        self.assertEqual(second.indices_existing, 21)
        self.assertEqual(second.constraints_created, 0)
        self.assertEqual(second.constraints_existing, 25)

    def test_does_not_hide_unexpected_index_errors(self) -> None:
        class BrokenGraph(FakeGraph):
            def _index(self, *args: object, **kwargs: object):
                raise RuntimeError("connection lost")

        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            SchemaManager(BrokenGraph()).ensure()


if __name__ == "__main__":
    unittest.main()
