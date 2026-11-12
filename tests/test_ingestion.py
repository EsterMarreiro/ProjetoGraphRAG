from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock

from licenciagraph.ingestion.curated import (
    activity_documents,
    checklist_documents,
    cost_documents,
    flow_document,
    merge_activities,
    prose_documents,
)
from licenciagraph.ingestion.facts import accepted_name
from licenciagraph.ingestion.models import ExtractedEntity, ExtractionResult
from licenciagraph.ingestion.pipeline import IngestionPipeline
from licenciagraph.ingestion.writer import GraphWriter
from licenciagraph.sources.documents import item_key

ITEM_TEXT = "Anotação de Responsabilidade Técnica (ART) dos projetos"


def sample_data():
    return {
        "manual": {
            "checklists": [
                {
                    "key": "AG-02",
                    "code": "AG-02",
                    "family": "AG",
                    "modality": "LI",
                    "variant": "",
                    "heading": ["LICENÇA DE INSTALAÇÃO"],
                    "pages": [101, 101],
                    "items": [
                        {"number": 1, "text": "Requerimento padrão SUDEMA"},
                        {"number": 2, "text": ITEM_TEXT},
                    ],
                    "notes": [],
                    "annex": [],
                }
            ],
            "activities": [
                {
                    "code": "16.15.599",
                    "name": "Avicultura industrial (postura e corte)",
                    "group": ["16", "AGROPECUÁRIA"],
                    "subgroup": ["16.15", "AVICULTURA"],
                    "regime": "licenca",
                    "page": 98,
                    "checklists_by_modality": {"LI": ["AG-02"]},
                    "checklists": ["AG-02"],
                }
            ],
            "generic_checklists": {"group": "72", "by_modality": {}},
        },
        "na101": {
            "master": [],
            "listed_activities": [
                {
                    "regime": "lau",
                    "code": "49.65.360",
                    "name": "Avicultura industrial acima de 210.000",
                    "validity": "5 ANOS",
                    "letter": "F",
                    "page": 33,
                }
            ],
            "porte_tables": [
                {
                    "codes": ["16.15.599"],
                    "alt_codes": {"LAU": "49.65.360"},
                    "name": "Avicultura industrial",
                    "criterion_label": "Número de cabeças",
                    "pages": [77],
                    "rows": [
                        {
                            "porte": "MEDIO",
                            "criterion": ">210.000 ≤ 840.000",
                            "letter": "H",
                            "letter_class": "MEDIO",
                            "annotation": "(LAU)",
                            "air": "M",
                            "water": "M",
                            "soil": "M",
                        }
                    ],
                }
            ],
            "cost_letters_ufr": {"H": {"LP": "21", "LAU": "16", "REGISTRO": "ISENTO"}},
            "base_criteria": {
                "I": [
                    {
                        "porte": "MICRO",
                        "area_m2": "Até 250",
                        "faturamento_ufr": "Até 7.500",
                        "funcionarios": "Até 10",
                    }
                ]
            },
        },
        "documentos": {
            item_key(ITEM_TEXT): {
                "name": "Anotação de Responsabilidade Técnica (ART)",
                "category": "responsabilidade_tecnica",
                "agency": "Conselho profissional",
                "condition": "",
                "studies": [],
            }
        },
    }


def relations(document):
    found = set()
    for planned in document.chunks:
        if planned.extraction is None:
            continue
        labels = {e.ref: e.label for e in planned.extraction.entities}
        for r in planned.extraction.relationships:
            found.add((labels[r.source_ref], r.relationship_type, labels[r.target_ref]))
    return found


class CuratedBuilderTests(unittest.TestCase):
    def test_checklist_chunk_links_documents_and_agencies(self):
        document = checklist_documents(sample_data())
        self.assertEqual(len(document.chunks), 1)
        text = document.chunks[0].chunk.text
        self.assertIn("Checklist AG-02", text)
        self.assertIn("1. Requerimento padrão SUDEMA", text)
        self.assertEqual(document.chunks[0].chunk.page, 101)
        found = relations(document)
        self.assertIn(("Checklist", "DO_TIPO", "Modalidade"), found)
        self.assertIn(("Checklist", "EXIGE_DOCUMENTO", "DocumentoExigido"), found)
        self.assertIn(("DocumentoExigido", "EMITIDO_POR", "Orgao"), found)

    def test_unnormalized_items_do_not_create_documents(self):
        document = checklist_documents(sample_data())
        labels = [e.label for e in document.chunks[0].extraction.entities]
        self.assertEqual(labels.count("DocumentoExigido"), 1)

    def test_activity_sheet_merges_regimes_porte_and_checklists(self):
        data = sample_data()
        document = activity_documents(data)
        self.assertEqual(len(document.chunks), 2)
        sheet = next(c for c in document.chunks if "16.15.599" in c.chunk.text)
        text = sheet.chunk.text
        self.assertIn("Licença Ambiental Única (LAU)", text)
        self.assertIn("validade de 5 anos", text)
        self.assertIn("checklist AG-02", text)
        self.assertIn("letra H", text)
        self.assertIn("Fontes: Manual p. 98", text)
        found = relations(document)
        for expected in (
            ("Atividade", "ENQUADRADA_EM", "Modalidade"),
            ("Atividade", "EXIGE_CHECKLIST", "Checklist"),
            ("Atividade", "TEM_FAIXA", "Faixa"),
            ("Faixa", "DE_PORTE", "Porte"),
            ("Faixa", "USA_LETRA", "LetraCusto"),
            ("Subgrupo", "SUBGRUPO_DE", "Grupo"),
        ):
            self.assertIn(expected, found)

    def test_activity_names_prefer_the_manual(self):
        merged = merge_activities(sample_data())
        self.assertEqual(
            merged["16.15.599"]["name"], "Avicultura industrial (postura e corte)"
        )
        self.assertEqual(merged["16.15.599"]["regimes"]["lau"]["letter"], "F")

    def test_cost_matrix_creates_cost_edges_with_ufr_values(self):
        document = cost_documents(sample_data())
        extraction = document.chunks[0].extraction
        values = {
            r.target_ref: r.properties["valor_ufr"]
            for r in extraction.relationships
            if r.relationship_type == "CUSTA"
        }
        self.assertEqual(values["modalidade:lau"], 16)
        self.assertEqual(values["modalidade:registro"], 0)
        self.assertIn("Letra H", document.chunks[0].chunk.text)

    def test_flow_document_encodes_license_sequence(self):
        extraction = flow_document().chunks[0].extraction
        pairs = {(r.source_ref, r.target_ref) for r in extraction.relationships}
        self.assertIn(("modalidade:lp", "modalidade:li"), pairs)
        self.assertIn(("modalidade:li", "modalidade:lo"), pairs)

    def test_prose_documents_are_extracted_by_llm_later(self):
        directory = Path(__file__).resolve().parents[1] / "data" / "curated"
        documents = prose_documents(directory)
        self.assertTrue(documents)
        self.assertTrue(all(c.extraction is None for d in documents for c in d.chunks))


class FactVocabularyTests(unittest.TestCase):
    def test_rejects_values_outside_the_vocabulary(self):
        self.assertEqual(accepted_name("Orgao", "SUDEMA"), "SUDEMA")
        self.assertIsNone(accepted_name("Orgao", "UFR-PB"))
        self.assertIsNone(accepted_name("Orgao", "órgão ambiental"))
        self.assertEqual(accepted_name("Estudo", "EIA"), "EIA/RIMA")
        self.assertIsNone(accepted_name("Norma", "Manual de Licenciamento Ambiental"))
        self.assertEqual(
            accepted_name("Norma", "Resolução CONAMA nº 237/1997"),
            "Resolução CONAMA nº 237/1997",
        )


class RecordingGraph:
    def __init__(self):
        self.queries = []

    def query(self, text, params=None):
        self.queries.append((text, params or {}))
        return Mock(result_set=[[0]])

    def ro_query(self, text, params=None):
        return Mock(result_set=[[0]])


class WriterTests(unittest.TestCase):
    def entity(self):
        return ExtractedEntity(
            "e", "Faixa", "Faixa X", {"letra": "H", "desconhecida": 1}, "faixa:x"
        )

    def test_entity_without_embedding_leaves_the_vector_untouched(self):
        graph = RecordingGraph()
        GraphWriter(graph).upsert_entity(self.entity(), None)
        text, params = graph.queries[-1]
        self.assertNotIn("embedding", text)
        self.assertEqual(params["letra"], "H")
        self.assertNotIn("desconhecida", params)

    def test_entity_with_embedding_sets_the_vector(self):
        graph = RecordingGraph()
        GraphWriter(graph).upsert_entity(self.entity(), [0.1, 0.2])
        self.assertIn("vecf32($embedding)", graph.queries[-1][0])

    def test_orphan_cleanup_never_touches_provenance(self):
        graph = RecordingGraph()
        GraphWriter(graph).remove_orphans()
        text = graph.queries[-1][0]
        self.assertIn("NOT n:Trecho AND NOT n:Documento", text)


class PipelineTests(unittest.TestCase):
    def test_embeds_chunks_and_only_vectorized_entity_labels_once(self):
        document = activity_documents(sample_data())
        llm = Mock()
        llm.embed.side_effect = lambda **kw: [[0.0] * 2 for _ in kw["texts"]]
        writer = Mock()
        writer.is_document_current.return_value = False
        report = IngestionPipeline(
            writer=writer,
            extractor=None,
            llm=llm,
            embedding_model="fake",
            embedding_dimension=2,
        ).run([document])
        self.assertEqual(report.documents_processed, 1)
        self.assertEqual(report.chunks_processed, 2)
        embedded = {
            call.args[0].canonical_id
            for call in writer.upsert_entity.call_args_list
            if call.args[1] is not None
        }
        self.assertIn("atividade:16.15.599", embedded)
        self.assertFalse(any(i.startswith("faixa:") for i in embedded))
        self.assertFalse(any(i.startswith("checklist:") for i in embedded))
        writer.complete_document.assert_called_once()
        writer.remove_orphans.assert_called_once()

    def test_skips_documents_that_are_already_current(self):
        writer = Mock()
        writer.is_document_current.return_value = True
        report = IngestionPipeline(
            writer=writer,
            extractor=None,
            llm=Mock(),
            embedding_model="fake",
            embedding_dimension=2,
        ).run([flow_document()])
        self.assertEqual(report.documents_skipped, 1)
        writer.prepare_document.assert_not_called()

    def test_prose_chunks_use_the_injected_extractor(self):
        directory = Path(__file__).resolve().parents[1] / "data" / "curated"
        document = prose_documents(directory)[0]
        document.chunks = document.chunks[:1]
        extractor = Mock()
        extractor.extract.return_value = ExtractionResult()
        llm = Mock()
        llm.embed.side_effect = lambda **kw: [[0.0] * 2 for _ in kw["texts"]]
        writer = Mock()
        writer.is_document_current.return_value = False
        IngestionPipeline(
            writer=writer,
            extractor=extractor,
            llm=llm,
            embedding_model="fake",
            embedding_dimension=2,
        ).run([document])
        extractor.extract.assert_called_once()


if __name__ == "__main__":
    unittest.main()
