from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from licenciagraph.config import Settings
from licenciagraph.retrieval.models import (
    GraphEdge,
    GraphNode,
    GroundedAnswer,
    RetrievalResult,
    SearchHit,
)
from licenciagraph.ui.runtime import UIRuntime

ROOT = Path(__file__).resolve().parents[1]


def fake_runtime(answer: GroundedAnswer | Exception) -> UIRuntime:
    settings = Settings(openai_api_key="k")
    llm = Mock()
    llm.list_models.return_value = [settings.chat_model, settings.embedding_model]
    service = Mock()
    if isinstance(answer, Exception):
        service.answer.side_effect = answer
    else:
        service.answer.return_value = answer
    return UIRuntime(settings, Mock(), llm, service)


def sample_answer() -> GroundedAnswer:
    retrieval = RetrievalResult(
        question="q",
        hits=[SearchHit("t1", "Texto.", "d1", "Curso A", "a.txt", 1.0, {"vector"})],
        nodes=[GraphNode("a", "Curso", "A"), GraphNode("b", "Curso", "B")],
        edges=[GraphEdge("a", "b", "PRE_REQUISITO", 1.0, "t1")],
        queries=["MATCH (n) RETURN n"],
        confidence=0.9,
        paths=[["A", "B"]],
        communities=[["A", "B"]],
    )
    return GroundedAnswer("A exige B. [F1]", ["F1"], ["aviso"], retrieval, ["G1"])


class StreamlitAppTests(unittest.TestCase):
    def _run(self, answer):
        runtime = fake_runtime(answer)
        with patch("licenciagraph.ui.app.get_runtime", return_value=runtime):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
            app.run()
            app.chat_input[0].set_value("Quem exige B?").run()
        return app

    def test_answer_is_rendered_with_trace_and_without_exceptions(self) -> None:
        app = self._run(sample_answer())
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 2)
        self.assertIn("A exige B", app.chat_message[1].markdown[0].value)
        text = " ".join(item.value for item in app.markdown)
        self.assertIn("Caminho mais curto por pré-requisitos", text)
        self.assertIn("Comunidades de cursos (Louvain)", text)

    def test_service_failure_is_shown_to_the_user(self) -> None:
        app = self._run(RuntimeError("falha simulada"))
        self.assertFalse(app.exception)
        self.assertTrue(any("falha simulada" in e.value for e in app.error))


if __name__ == "__main__":
    unittest.main()
