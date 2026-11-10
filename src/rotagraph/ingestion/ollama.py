"""Cliente HTTP minimo para a API local do Ollama."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaClient:
    def __init__(self, base_url: str, *, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat_json(
        self, *, model: str, system: str, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        response = self._post("/api/chat", payload)
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Resposta do Ollama nao contem message.content")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Resposta estruturada do Ollama deve ser um objeto JSON")
        return parsed

    def list_models(self) -> list[str]:
        response = self._get("/api/tags")
        models = response.get("models", [])
        if not isinstance(models, list):
            raise ValueError("Ollama retornou uma lista de modelos invalida")
        return [
            str(item.get("name") or item.get("model"))
            for item in models
            if isinstance(item, dict) and (item.get("name") or item.get("model"))
        ]

    def embed(
        self, *, model: str, texts: list[str], dimensions: int
    ) -> list[list[float]]:
        if not texts:
            return []
        response = self._post(
            "/api/embed",
            {
                "model": model,
                "input": texts,
                "dimensions": dimensions,
                "truncate": True,
            },
        )
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ValueError(
                "Quantidade de embeddings diferente da quantidade de textos"
            )
        for vector in embeddings:
            if not isinstance(vector, list) or len(vector) != dimensions:
                raise ValueError("Dimensao de embedding diferente da configurada")
        return embeddings

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama retornou HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Nao foi possivel conectar ao Ollama: {exc.reason}"
            ) from exc
        if not isinstance(result, dict):
            raise ValueError("Ollama retornou um JSON inesperado")
        return result

    def _get(self, path: str) -> dict[str, Any]:
        request = Request(f"{self.base_url}{path}", method="GET")
        try:
            with urlopen(request, timeout=min(self.timeout, 5)) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama retornou HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Nao foi possivel conectar ao Ollama: {exc.reason}"
            ) from exc
        if not isinstance(result, dict):
            raise ValueError("Ollama retornou um JSON inesperado")
        return result
