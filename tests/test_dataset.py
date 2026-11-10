from __future__ import annotations

import re
import unittest
from pathlib import Path

from rotagraph.ingestion.loader import load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_DOC = PROJECT_ROOT / "docs" / "03-dataset.md"


class TourismDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = load_documents(DATA_DIR)
        cls.corpus = "\n".join(document.content for document in cls.documents)

    def test_has_expected_number_of_documents(self) -> None:
        self.assertGreaterEqual(len(self.documents), 10)
        self.assertLessEqual(len(self.documents), 15)

    def test_every_document_has_required_header_and_route(self) -> None:
        required = (
            "Titulo:",
            "Regiao:",
            "UF:",
            "Data de referencia do corpus:",
        )
        for document in self.documents:
            with self.subTest(document=document.path.name):
                for field in required:
                    self.assertIn(field, document.content)
                self.assertRegex(document.content, r"A Rota .+ inclui, (?:em|na) ordem")

    def test_covers_all_brazilian_regions(self) -> None:
        for region in ("Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"):
            with self.subTest(region=region):
                self.assertIn(f"Regiao: {region}", self.corpus)

    def test_documents_cover_safety_and_provenance_language(self) -> None:
        for document in self.documents:
            with self.subTest(document=document.path.name):
                normalized = document.content.casefold()
                self.assertIn("adequad", normalized)
                self.assertIn("recomendad", normalized)
                self.assertIn("evitar", normalized)
                self.assertIn("confirm", normalized)

    def test_evaluation_document_has_at_least_twelve_questions(self) -> None:
        content = DATASET_DOC.read_text(encoding="utf-8")
        questions = re.findall(r"^\d+\. \*\*", content, flags=re.MULTILINE)
        self.assertGreaterEqual(len(questions), 12)


if __name__ == "__main__":
    unittest.main()
