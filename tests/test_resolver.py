from __future__ import annotations

import unittest
from dataclasses import replace

from licenciagraph.retrieval.generation import (
    SIMILARITY_NOTICE,
    _build_context,
    _with_resolution_notes,
)
from licenciagraph.retrieval.models import (
    GraphNode,
    GroundedAnswer,
    Resolution,
    RetrievalResult,
    SearchHit,
)
from licenciagraph.retrieval.resolver import ActivityResolver, stems


class FakeLLM:
    def __init__(self, interpret: dict, choose: dict) -> None:
        self.interpret = interpret
        self.choose = choose
        self.prompts: list[str] = []

    def chat_json(self, *, system, prompt, **_):
        self.prompts.append(prompt)
        return self.interpret if "historico" in prompt else self.choose

    def embed(self, *, texts, **_):
        return [[float(i)] for i, _ in enumerate(texts)]


class FakeRepository:
    def __init__(self) -> None:
        self.stems: list[str] = []

    def activity_scores(self, vector, *, limit):
        return [("a:1", 0.8), ("a:2", 0.7)]

    def activity_ids_by_stem(self, stems):
        self.stems = stems
        return ["a:3"]

    def activity_details(self, ids):
        names = {"a:1": "Suinocultura", "a:2": "Avicultura", "a:3": "Frigorífico"}
        return {
            i: {"code": i, "name": names[i], "group": "AGRO", "regimes": ["LAC"]}
            for i in ids
        }


def resolver(interpret: dict, choose: dict) -> tuple[ActivityResolver, FakeLLM]:
    llm = FakeLLM(interpret, choose)
    return (
        ActivityResolver(
            llm,
            FakeRepository(),
            model="m",
            embedding_model="e",
            embedding_dimension=1,
        ),
        llm,
    )


class ResolverTests(unittest.TestCase):
    def test_colloquial_question_is_mapped_to_listed_activities(self) -> None:
        r, _ = resolver(
            {
                "pergunta_autonoma": "vender porcos",
                "tem_atividade": True,
                "termos": ["suinocultura", "criação de suínos"],
            },
            {
                "tipo": "exata",
                "escolhidas": [1],
                "motivo": "m",
                "faltam": ["nº animais"],
            },
        )
        result = r.resolve("vender porcos")
        self.assertEqual(result.match, "exata")
        self.assertEqual([n.name for n in result.activities], ["Suinocultura"])
        self.assertEqual(result.missing, ("nº animais",))

    def test_general_question_skips_candidate_search(self) -> None:
        r, llm = resolver(
            {"pergunta_autonoma": "o que é LAC?", "tem_atividade": False, "termos": []},
            {},
        )
        result = r.resolve("o que é LAC?")
        self.assertEqual(result.match, "nenhuma")
        self.assertEqual(len(llm.prompts), 1)

    def test_choice_outside_the_candidate_list_is_discarded(self) -> None:
        r, _ = resolver(
            {"pergunta_autonoma": "x", "tem_atividade": True, "termos": ["a"]},
            {"tipo": "proxima", "escolhidas": [99], "motivo": "", "faltam": ["y"]},
        )
        result = r.resolve("x")
        self.assertEqual(result.match, "nenhuma")
        self.assertEqual(result.activities, ())
        self.assertEqual(result.missing, ())

    def test_other_jurisdiction_is_flagged_without_candidate_search(self) -> None:
        r, llm = resolver(
            {
                "pergunta_autonoma": "documentos da CETESB",
                "fora_do_escopo": True,
                "tem_atividade": True,
                "termos": ["fábrica"],
            },
            {},
        )
        result = r.resolve("documentos da CETESB")
        self.assertEqual(result.match, "fora_do_escopo")
        self.assertEqual(result.activities, ())
        self.assertEqual(len(llm.prompts), 1)

    def test_history_is_sent_to_the_interpreter(self) -> None:
        r, llm = resolver(
            {"pergunta_autonoma": "custo", "tem_atividade": False, "termos": []}, {}
        )
        r.resolve("e o custo?", [("Vou criar porcos", "Depende do número")])
        self.assertIn("Vou criar porcos", llm.prompts[0])

    def test_stems_skip_generic_words(self) -> None:
        self.assertEqual(stems(("atividades de suinocultura",)), ["suinocult"])
        self.assertEqual(stems(("serviços gerais",)), ["gerais"[:5]])


class ResolutionNotesTests(unittest.TestCase):
    def _answer(self, resolution: Resolution, abstained=False) -> GroundedAnswer:
        retrieval = RetrievalResult(
            "q",
            [SearchHit("t", "x", "d", "T", "f", 1.0)],
            [],
            [],
            [],
            0.5,
            resolution=resolution,
        )
        return GroundedAnswer(
            "Resposta. [F1]", ["F1"], [], retrieval, abstained=abstained
        )

    def test_similarity_notice_and_missing_data_are_appended(self) -> None:
        node = GraphNode("a", "Atividade", "A")
        answer = self._answer(Resolution("proxima", (node,), "", ("área útil",)))
        text = _with_resolution_notes(answer, answer.retrieval).answer
        self.assertIn(SIMILARITY_NOTICE, text)
        self.assertIn("informe: área útil.", text)

    def test_exact_match_only_asks_for_missing_data(self) -> None:
        node = GraphNode("a", "Atividade", "A")
        answer = self._answer(Resolution("exata", (node,), "", ()))
        self.assertEqual(
            _with_resolution_notes(answer, answer.retrieval).answer, answer.answer
        )

    def test_abstention_is_left_untouched(self) -> None:
        node = GraphNode("a", "Atividade", "A")
        answer = self._answer(Resolution("proxima", (node,)), abstained=True)
        self.assertEqual(
            _with_resolution_notes(answer, answer.retrieval).answer, answer.answer
        )

    def test_context_does_not_forward_the_model_rationale(self) -> None:
        node = GraphNode("a", "Atividade", "A")
        resolution = Resolution("proxima", (node,), "gera perfurocortantes", ())
        context = _build_context(
            replace(self._answer(resolution).retrieval, resolution=resolution)
        )
        self.assertIn("ENQUADRAMENTO: tipo=proxima", context)
        self.assertNotIn("perfurocortantes", context)
