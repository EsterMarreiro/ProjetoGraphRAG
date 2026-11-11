from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trilhagraph.ingestion.canonicalization import (
    EntityCanonicalizer,
    normalize_text,
    slugify,
)
from trilhagraph.ingestion.chunking import split_document
from trilhagraph.ingestion.extraction import parse_extraction
from trilhagraph.ingestion.loader import load_documents
from trilhagraph.ingestion.models import Document, ExtractedEntity, ExtractionResult
from trilhagraph.ingestion.pipeline import IngestionPipeline


class LoaderAndChunkingTests(unittest.TestCase):
    def test_loads_only_non_empty_txt_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "destino.txt").write_text(
                "# Programação\n\nProgramação fica na Bahia.", encoding="utf-8"
            )
            (data_dir / "vazio.txt").write_text("", encoding="utf-8")
            (data_dir / "ignorar.md").write_text("texto", encoding="utf-8")

            documents = load_documents(data_dir)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].title, "Programação")
        self.assertEqual(documents[0].id, "documento:destino")
        self.assertEqual(len(documents[0].checksum), 64)

    def test_chunk_ids_are_stable_and_ordered(self) -> None:
        document = Document(
            id="documento:teste",
            title="Teste",
            path=Path("teste.txt"),
            checksum="x",
            content="Primeiro paragrafo.\n\nSegundo paragrafo mais longo.",
        )
        chunks = split_document(document, chunk_size=30, overlap=5)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].id, "documento:teste:trecho:0001")
        positions = [item.position for item in chunks]
        self.assertEqual(positions, list(range(1, len(chunks) + 1)))
        self.assertTrue(all(len(item.text) <= 30 for item in chunks))


class CanonicalizationTests(unittest.TestCase):
    def test_normalizes_accents_and_aliases(self) -> None:
        canonicalizer = EntityCanonicalizer()
        first = ExtractedEntity(
            "a",
            "Area",
            "Programação Python",
            {"aliases": ["Python"], "provedor": "Equipe A"},
        )
        alias = ExtractedEntity("b", "Area", "python")

        first_id = canonicalizer.canonicalize(first)
        alias_id = canonicalizer.canonicalize(alias)

        self.assertEqual(normalize_text("  Programação  Python "), "programacao python")
        self.assertEqual(slugify("Recuperação Híbrida"), "recuperacao-hibrida")
        self.assertEqual(first_id, alias_id)

    def test_does_not_merge_homonyms_from_different_providers(self) -> None:
        canonicalizer = EntityCanonicalizer()
        first = ExtractedEntity("a", "Area", "Curso Homônimo", {"provedor": "Equipe A"})
        second = ExtractedEntity(
            "b", "Area", "Curso Homônimo", {"provedor": "Equipe B"}
        )

        self.assertNotEqual(
            canonicalizer.canonicalize(first), canonicalizer.canonicalize(second)
        )


class ExtractionParsingTests(unittest.TestCase):
    def test_parses_valid_graph(self) -> None:
        result = parse_extraction(
            {
                "entities": [
                    {
                        "ref": "a1",
                        "label": "Curso",
                        "name": "APIs Web",
                        "properties": {"tipo": "historica"},
                    },
                    {
                        "ref": "l1",
                        "label": "Area",
                        "name": "Programação",
                        "properties": {"tipo": "fundamentos"},
                    },
                ],
                "relationships": [
                    {
                        "source_ref": "a1",
                        "target_ref": "l1",
                        "relationship_type": "ABORDA",
                        "confidence": 0.98,
                        "properties": {},
                    }
                ],
            }
        )

        self.assertEqual(len(result.entities), 2)
        self.assertEqual(result.relationships[0].relationship_type, "ABORDA")

    def test_rejects_dangling_relationship(self) -> None:
        raw = {
            "entities": [{"ref": "a", "label": "Curso", "name": "X", "properties": {}}],
            "relationships": [
                {
                    "source_ref": "a",
                    "target_ref": "missing",
                    "relationship_type": "ABORDA",
                    "confidence": 1,
                    "properties": {},
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "referencia inexistente"):
            parse_extraction(raw)


class FakeWriter:
    def __init__(self) -> None:
        self.current = False
        self.documents = []
        self.chunks = []
        self.entities = []
        self.relationships = []

    def is_document_current(self, document: Document) -> bool:
        return self.current

    def prepare_document(self, document: Document) -> None:
        self.current = False
        self.documents.append(document)

    def complete_document(self, document: Document) -> None:
        self.current = True

    def upsert_chunk(self, chunk, embedding) -> None:
        self.chunks.append((chunk, embedding))

    def upsert_entity(self, entity, embedding) -> None:
        self.entities.append((entity, embedding))

    def link_mention(self, chunk_id, entity) -> None:
        pass

    def upsert_relationship(self, relationship, entities_by_ref, *, chunk_id) -> None:
        self.relationships.append((relationship, chunk_id))


class FakeExtractor:
    def extract(self, text: str) -> ExtractionResult:
        return ExtractionResult(
            entities=[
                ExtractedEntity("l1", "Area", "Programação", {"tipo": "fundamentos"})
            ]
        )


class FakeLLM:
    def embed(self, *, model: str, texts: list[str], dimensions: int):
        return [[0.0] * dimensions for _ in texts]


class PipelineTests(unittest.TestCase):
    def test_processes_document_and_skips_unchanged_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "programacao.txt").write_text(
                "Programação\n\nProgramação oferece aprendizagem.", encoding="utf-8"
            )
            writer = FakeWriter()
            pipeline = IngestionPipeline(
                writer=writer,
                extractor=FakeExtractor(),
                llm=FakeLLM(),
                embedding_model="fake",
                embedding_dimension=4,
                chunk_size=100,
                chunk_overlap=10,
            )

            first = pipeline.run(data_dir)
            writer.current = True
            second = pipeline.run(data_dir)

        self.assertEqual(first.documents_processed, 1)
        self.assertEqual(first.entities_processed, 1)
        self.assertEqual(len(writer.chunks[0][1]), 4)
        self.assertEqual(second.documents_skipped, 1)


if __name__ == "__main__":
    unittest.main()
