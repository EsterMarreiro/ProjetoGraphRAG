from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from licenciagraph.evaluation.evaluator import (
    EvaluationCase,
    evaluate_service,
    load_cases,
    render_markdown,
)
from licenciagraph.retrieval.models import (
    GraphEdge,
    GroundedAnswer,
    RetrievalResult,
    SearchHit,
)


class FakeService:
    def answer(self, question: str) -> GroundedAnswer:
        hit = SearchHit(
            chunk_id="trecho:1",
            text="O checklist AG-02 exige a ART e o CAR.",
            document_id="doc:1",
            title="Manual de Licenciamento Ambiental",
            filename="manual.pdf",
            score=0.9,
            channels={"vector", "fulltext"},
        )
        retrieval = RetrievalResult(
            question=question,
            hits=[hit],
            nodes=[],
            edges=[
                GraphEdge("atividade:1", "checklist:1", "EXIGE_CHECKLIST"),
                GraphEdge("checklist:1", "modalidade:li", "DO_TIPO"),
            ],
            queries=["MATCH ..."],
            confidence=0.8,
        )
        return GroundedAnswer(
            answer="O checklist AG-02 exige a ART e o CAR [F1].",
            citations=["F1"],
            caveats=[],
            retrieval=retrieval,
            supporting_edges=["G1", "G2"],
        )


class EvaluationTests(unittest.TestCase):
    def test_scores_sources_concepts_citations_and_graph(self) -> None:
        case = EvaluationCase(
            id="Q1",
            question="Que documentos o checklist exige?",
            expected_sources=("trecho:1",),
            expected_concepts=("AG-02", "ART"),
            requires_multihop=True,
        )

        report = evaluate_service(FakeService(), [case])

        self.assertEqual(report.pass_rate, 1.0)
        self.assertEqual(report.mean_source_recall, 1.0)
        self.assertEqual(report.mean_concept_coverage, 1.0)
        self.assertEqual(report.multihop_context_rate, 1.0)
        self.assertIn("| Q1 | OK |", render_markdown(report))

    def test_loads_and_validates_json_dataset(self) -> None:
        data = [
            {
                "id": "Q1",
                "question": "Pergunta",
                "expected_sources": ["trecho:1"],
                "expected_concepts": ["conceito"],
            }
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            cases = load_cases(path)

        self.assertEqual(cases[0].id, "Q1")
        self.assertFalse(cases[0].requires_multihop)


if __name__ == "__main__":
    unittest.main()
