from __future__ import annotations

import unittest
from unittest.mock import Mock

from licenciagraph.evaluation.judge import Judge, summarize


def claim(text, verdict="supported"):
    return {"claim": text, "quote": "trecho", "verdict": verdict}


class JudgeTests(unittest.TestCase):
    def _judge(self, response):
        client = Mock()
        client.chat_json.return_value = response
        return Judge(client, model="j", corpus="corpus"), client

    def test_correct_needs_all_facts_no_false_claims_and_answered(self) -> None:
        judge, client = self._judge(
            {
                "claims": [claim("A")],
                "facts": [{"fact": "x", "stated_correctly": True}],
                "answers_question": True,
            }
        )
        verdict = judge.judge_positive("Q1", "p?", "A [F1]", ["x"])
        self.assertTrue(verdict.correct)
        self.assertNotIn("[F1]", client.chat_json.call_args.kwargs["prompt"])
        self.assertIn("corpus", client.chat_json.call_args.kwargs["prompt"])

    def test_contradicted_or_unverifiable_claims_are_false(self) -> None:
        judge, _ = self._judge(
            {
                "claims": [claim("ok"), claim("inventada", "unverifiable")],
                "facts": [{"fact": "x", "stated_correctly": True}],
                "answers_question": True,
            }
        )
        verdict = judge.judge_positive("Q1", "p?", "texto", ["x"])
        self.assertFalse(verdict.correct)
        self.assertEqual(verdict.false_claims, ("inventada",))

    def test_missing_fact_makes_answer_incomplete(self) -> None:
        judge, _ = self._judge(
            {
                "claims": [],
                "facts": [{"fact": "x", "stated_correctly": False}],
                "answers_question": True,
            }
        )
        verdict = judge.judge_positive("Q1", "p?", "texto", ["x"])
        self.assertFalse(verdict.complete)

    def test_negative_must_decline_without_fabrication(self) -> None:
        judge, _ = self._judge({"claims": [], "declines_or_corrects": True})
        self.assertTrue(judge.judge_negative("N1", "p?", "não consta", "x").correct)
        judge, _ = self._judge(
            {
                "claims": [claim("inventou", "contradicted")],
                "declines_or_corrects": True,
            }
        )
        self.assertFalse(judge.judge_negative("N1", "p?", "inventou", "x").correct)

    def test_summary_counts_hallucinations(self) -> None:
        judge, _ = self._judge(
            {
                "claims": [claim("falsa", "contradicted")],
                "facts": [{"fact": "x", "stated_correctly": True}],
                "answers_question": True,
            }
        )
        summary = summarize([judge.judge_positive("Q1", "p", "a", ["x"])])
        self.assertEqual(summary["with_false_claims"], 1)
        self.assertEqual(summary["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
