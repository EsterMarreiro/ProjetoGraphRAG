"""Cliente HTTP minimo para a API da OpenAI (chat estruturado e embeddings)."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY nao configurada")
        self._api_key = api_key
        self.usage: dict[str, dict[str, int]] = {}
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.chat_options = {"temperature": 0, "seed": 42, "max_tokens": 2048}

    @classmethod
    def from_settings(cls, settings: Any) -> OpenAIClient:
        return cls(
            settings.openai_api_key or "",
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout,
        )

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        strict: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            **self.chat_options,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "resposta", "schema": schema, "strict": strict},
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        response = self._request("POST", "/chat/completions", payload)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message", {})
        content = message.get("content")
        if message.get("refusal") or not isinstance(content, str):
            raise ValueError("Resposta da OpenAI sem conteudo estruturado")
        if choice.get("finish_reason") == "length":
            raise ValueError("Resposta da OpenAI truncada por max_tokens")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Resposta estruturada da OpenAI deve ser um objeto JSON")
        return parsed

    def embed(
        self, *, model: str, texts: list[str], dimensions: int
    ) -> list[list[float]]:
        if not texts:
            return []
        response = self._request(
            "POST",
            "/embeddings",
            {"model": model, "input": texts, "dimensions": dimensions},
        )
        data = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding") for item in data]
        if len(embeddings) != len(texts):
            raise ValueError(
                "Quantidade de embeddings diferente da quantidade de textos"
            )
        for vector in embeddings:
            if not isinstance(vector, list) or len(vector) != dimensions:
                raise ValueError("Dimensao de embedding diferente da configurada")
        return embeddings

    def list_models(self) -> list[str]:
        response = self._request("GET", "/models", None, timeout=10)
        return [str(item["id"]) for item in response.get("data", []) if "id" in item]

    def _count_usage(self, payload: dict[str, Any] | None, result: dict[str, Any]):
        usage = result.get("usage") or {}
        model = str((payload or {}).get("model", "?"))
        entry = self.usage.setdefault(model, {"input": 0, "output": 0, "calls": 0})
        entry["input"] += int(
            usage.get("prompt_tokens") or usage.get("total_tokens") or 0
        )
        entry["output"] += int(usage.get("completion_tokens") or 0)
        entry["calls"] += 1

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            request = Request(
                f"{self.base_url}{path}",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method=method,
            )
            try:
                with urlopen(request, timeout=timeout or self.timeout) as response:
                    result = json.load(response)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code in RETRYABLE_STATUS and attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(
                    f"OpenAI retornou HTTP {exc.code}: {detail}"
                ) from exc
            except (URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(
                    f"Nao foi possivel conectar a OpenAI: {exc}"
                ) from exc
            if not isinstance(result, dict):
                raise ValueError("OpenAI retornou um JSON inesperado")
            self._count_usage(payload, result)
            return result
        raise AssertionError("Fluxo de requisicao inesperado")
