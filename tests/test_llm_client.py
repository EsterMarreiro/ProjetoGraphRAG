from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from trilhagraph.ingestion.llm import OpenAIClient


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def reply(payload: dict) -> FakeResponse:
    return FakeResponse(json.dumps(payload).encode())


class OpenAIClientTests(unittest.TestCase):
    def test_requires_key(self) -> None:
        with self.assertRaises(ValueError):
            OpenAIClient("")

    def test_chat_json_sends_schema_and_parses_content(self) -> None:
        captured = {}

        def fake(request, timeout):
            captured["body"] = json.loads(request.data)
            captured["auth"] = request.headers["Authorization"]
            return reply({"choices": [{"message": {"content": '{"r": "ok"}'}}]})

        with patch("trilhagraph.ingestion.llm.urlopen", fake):
            result = OpenAIClient("k").chat_json(
                model="m", system="s", prompt="p", schema={"type": "object"}
            )

        self.assertEqual(result, {"r": "ok"})
        self.assertEqual(captured["auth"], "Bearer k")
        self.assertEqual(captured["body"]["temperature"], 0)
        self.assertEqual(
            captured["body"]["response_format"]["json_schema"]["schema"],
            {"type": "object"},
        )

    def test_truncated_response_is_rejected(self) -> None:
        body = {"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]}
        with (
            patch("trilhagraph.ingestion.llm.urlopen", lambda *a, **k: reply(body)),
            self.assertRaisesRegex(ValueError, "truncada"),
        ):
            OpenAIClient("k").chat_json(model="m", system="s", prompt="p", schema={})

    def test_embeddings_are_ordered_and_validated(self) -> None:
        body = {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }
        with patch("trilhagraph.ingestion.llm.urlopen", lambda *a, **k: reply(body)):
            vectors = OpenAIClient("k").embed(model="m", texts=["a", "b"], dimensions=2)
            self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
            with self.assertRaisesRegex(ValueError, "Dimensao"):
                OpenAIClient("k").embed(model="m", texts=["a", "b"], dimensions=3)

    def test_retries_transient_errors(self) -> None:
        calls = []

        def fake(request, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise HTTPError(request.full_url, 429, "rate", {}, io.BytesIO(b"{}"))
            return reply({"data": []})

        with (
            patch("trilhagraph.ingestion.llm.urlopen", fake),
            patch("trilhagraph.ingestion.llm.time.sleep"),
        ):
            self.assertEqual(OpenAIClient("k").list_models(), [])
        self.assertEqual(len(calls), 2)

    def test_error_detail_never_contains_key(self) -> None:
        def fake(request, timeout):
            raise HTTPError(request.full_url, 401, "no", {}, io.BytesIO(b"denied"))

        with (
            patch("trilhagraph.ingestion.llm.urlopen", fake),
            self.assertRaises(RuntimeError) as ctx,
        ):
            OpenAIClient("segredo").list_models()
        self.assertNotIn("segredo", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
