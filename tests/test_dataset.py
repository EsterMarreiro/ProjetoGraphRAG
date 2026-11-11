from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from trilhagraph.ingestion.loader import load_documents

ROOT = Path(__file__).resolve().parents[1]


class LearningDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = load_documents(ROOT / "data/raw")
        cls.cases = json.loads((ROOT / "data/evaluation/questions.json").read_text())

    def test_catalog_and_cases_have_complete_source_references(self):
        self.assertEqual(len(self.documents), 12)
        self.assertEqual(len(self.cases), 15)
        filenames = {doc.path.name for doc in self.documents}
        for case in self.cases:
            self.assertLessEqual(set(case["expected_sources"]), filenames)
            self.assertTrue(case["expected_concepts"])

    def test_catalog_discloses_fictional_origin_and_limits(self):
        for doc in self.documents:
            with self.subTest(file=doc.path.name):
                self.assertIn("fictício e autoral", doc.content)
                self.assertIn("Não há informações sobre preços", doc.content)
                self.assertNotIn("Título:", doc.title)

    def test_prerequisites_are_defined_and_acyclic(self):
        names = {doc.title for doc in self.documents}
        adjacency = {}
        for doc in self.documents:
            dependencies = re.findall(
                r"exige como pré-requisito o curso ([^.]+)\.", doc.content
            )
            self.assertLessEqual(set(dependencies), names)
            adjacency[doc.title] = dependencies

        def visit(name, stack):
            self.assertNotIn(name, stack, f"Ciclo: {stack} -> {name}")
            for dependency in adjacency[name]:
                visit(dependency, stack + [name])

        for name in names:
            visit(name, [])

    def test_multihop_cases_require_cross_document_evidence(self):
        multi = [case for case in self.cases if case["requires_multihop"]]
        self.assertGreaterEqual(len(multi), 12)
        self.assertGreaterEqual(sum(len(c["expected_sources"]) >= 2 for c in multi), 10)
