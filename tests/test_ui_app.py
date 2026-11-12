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
    guide = Mock()
    guide.search_activities.return_value = []
    if isinstance(answer, Exception):
        service.answer.side_effect = answer
    else:
        service.answer.return_value = answer
    return UIRuntime(settings, Mock(), llm, service, guide)


def sample_answer() -> GroundedAnswer:
    retrieval = RetrievalResult(
        question="q",
        hits=[SearchHit("t1", "Texto.", "d1", "Manual", "a.pdf", 1.0, {"vector"}, 14)],
        nodes=[GraphNode("a", "Atividade", "A"), GraphNode("b", "Checklist", "B")],
        edges=[GraphEdge("a", "b", "EXIGE_CHECKLIST", 1.0, "t1")],
        queries=["MATCH (n) RETURN n"],
        confidence=0.9,
        paths=[["A", "B"]],
        communities=[["A", "B"]],
    )
    return GroundedAnswer(
        "A exige o checklist B. [F1]", ["F1"], ["aviso"], retrieval, ["G1"]
    )


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
        self.assertIn("A exige o checklist B", app.chat_message[1].markdown[0].value)
        text = " ".join(item.value for item in app.markdown)
        self.assertIn("Sequência de licenças (caminho mais curto)", text)
        self.assertIn("Atividades com checklists em comum (Louvain)", text)

    def test_guide_tab_is_available_and_searches_the_graph(self) -> None:
        runtime = fake_runtime(sample_answer())
        with patch("licenciagraph.ui.app.get_runtime", return_value=runtime):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
            app.run()
            app.text_input(key="guide_query").set_value("avicultura").run()
        self.assertFalse(app.exception)
        runtime.guide.search_activities.assert_called_with("avicultura")
        self.assertTrue(
            any("Nenhuma atividade encontrada" in i.value for i in app.info)
        )

    def test_service_failure_is_shown_to_the_user(self) -> None:
        app = self._run(RuntimeError("falha simulada"))
        self.assertFalse(app.exception)
        self.assertTrue(any("falha simulada" in e.value for e in app.error))


if __name__ == "__main__":
    unittest.main()
