"""Geracao restrita ao contexto recuperado."""

from __future__ import annotations

from typing import Any

from rotagraph.ingestion.ollama import OllamaClient
from rotagraph.retrieval.models import GroundedAnswer, RetrievalResult

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["answer", "citations", "caveats"],
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
}

SYSTEM_PROMPT = """Voce e um assistente de turismo brasileiro fundamentado.
Responda somente com os trechos e fatos do grafo fornecidos. Cite fontes usando
exatamente os IDs [F1], [F2] etc. Nao invente preco, horario, clima, distancia ou
disponibilidade. Preserve ressalvas sazonais e de seguranca. Se as evidencias nao
forem suficientes, declare isso claramente em answer e use caveats."""


class AnswerGenerator:
    def __init__(self, client: OllamaClient, *, model: str) -> None:
        self.client = client
        self.model = model

    def generate(self, retrieval: RetrievalResult) -> GroundedAnswer:
        if not retrieval.hits:
            return GroundedAnswer(
                answer="Não encontrei evidências suficientes no corpus para responder.",
                citations=[],
                caveats=[
                    "A resposta foi interrompida por ausência de contexto recuperado."
                ],
                retrieval=retrieval,
            )

        context = _build_context(retrieval)
        raw = self.client.chat_json(
            model=self.model,
            system=SYSTEM_PROMPT,
            prompt=f"Pergunta: {retrieval.question}\n\nContexto:\n{context}",
            schema=ANSWER_SCHEMA,
        )
        answer = _required_string(raw, "answer")
        valid_citations = {f"F{index}" for index in range(1, len(retrieval.hits) + 1)}
        citations = [
            citation.removeprefix("[").removesuffix("]")
            for citation in _string_list(raw, "citations")
        ]
        if any(citation not in valid_citations for citation in citations):
            raise ValueError("A resposta citou uma fonte inexistente")
        return GroundedAnswer(
            answer=answer,
            citations=citations,
            caveats=_string_list(raw, "caveats"),
            retrieval=retrieval,
        )


def _build_context(retrieval: RetrievalResult) -> str:
    sources = [
        f"[F{index}] arquivo={hit.filename}; titulo={hit.title}; texto={hit.text}"
        for index, hit in enumerate(retrieval.hits, start=1)
    ]
    graph_facts = [
        f"{edge.source} -[{edge.relationship_type}]-> {edge.target} "
        f"(trecho={edge.chunk_id or 'nao informado'})"
        for edge in retrieval.edges
    ]
    return "\n".join([*sources, "FATOS DO SUBGRAFO:", *graph_facts])


def _required_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Campo de resposta invalido: {key}")
    return value.strip()


def _string_list(raw: dict[str, Any], key: str) -> list[str]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Campo de resposta deve ser lista de strings: {key}")
    return [item.strip() for item in value if item.strip()]
