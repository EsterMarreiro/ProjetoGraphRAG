"""Juiz LLM com acesso ao corpus completo, usado para medir acerto e alucinação."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from licenciagraph.ingestion.llm import OpenAIClient

CLAIMS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["claim", "quote", "verdict"],
        "properties": {
            "claim": {"type": "string"},
            "quote": {"type": "string"},
            "verdict": {
                "type": "string",
                "enum": ["supported", "contradicted", "unverifiable"],
            },
        },
        "additionalProperties": False,
    },
}

POSITIVE_SCHEMA = {
    "type": "object",
    "required": ["claims", "facts", "answers_question"],
    "properties": {
        "claims": CLAIMS_SCHEMA,
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["fact", "stated_correctly"],
                "properties": {
                    "fact": {"type": "string"},
                    "stated_correctly": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
        "answers_question": {"type": "boolean"},
    },
    "additionalProperties": False,
}

NEGATIVE_SCHEMA = {
    "type": "object",
    "required": ["claims", "declines_or_corrects"],
    "properties": {
        "claims": CLAIMS_SCHEMA,
        "declines_or_corrects": {"type": "boolean"},
    },
    "additionalProperties": False,
}

CLAIM_RULES = """Decomponha a resposta em afirmações atômicas sobre o catálogo (ignore
marcadores [F1], avisos de que o catálogo é fictício e frases de recusa). Para cada
afirmação, copie em quote a(s) frase(s) do corpus que a confirmam ou refutam; use
quote vazio se não houver. verdict: supported (o corpus confirma),
contradicted (o corpus afirma o contrário) ou unverifiable (o corpus não trata do
assunto). Afirmações que combinam fatos de várias frases do corpus são supported
se cada fato combinado é verdadeiro. Afirmar que o catálogo NÃO informa algo é
supported quando o corpus de fato não trata do assunto. Não marque como
problema o que o corpus afirma; só use contradicted ou unverifiable quando o
corpus não sustentar."""

POSITIVE_SYSTEM = f"""Você é um revisor rigoroso. Recebe o CORPUS completo de um
catálogo fictício, uma pergunta, uma resposta e fatos de referência.
{CLAIM_RULES}
Também diga, para cada fato de referência, se a resposta o afirma corretamente
(stated_correctly) e se a resposta atende ao que foi perguntado."""

NEGATIVE_SYSTEM = f"""Você é um revisor rigoroso. Recebe o CORPUS completo de um
catálogo fictício, uma pergunta cuja resposta NÃO está no corpus (ou que traz
premissa falsa) e o comportamento esperado.
{CLAIM_RULES}
declines_or_corrects é verdadeiro se a resposta admite a ausência da informação
ou corrige a premissa falsa."""


@dataclass(frozen=True, slots=True)
class Verdict:
    id: str
    correct: bool
    complete: bool
    false_claims: tuple[str, ...]
    detail: dict[str, Any]


class Judge:
    def __init__(self, client: OpenAIClient, *, model: str, corpus: str) -> None:
        self.client = client
        self.model = model
        self.corpus = corpus

    @classmethod
    def from_directory(cls, client: OpenAIClient, *, model: str, directory: Path):
        parts = [
            f"## {path.name}\n{path.read_text(encoding='utf-8')}"
            for path in sorted(directory.glob("*.txt"))
        ]
        return cls(client, model=model, corpus="\n\n".join(parts))

    @staticmethod
    def _false_claims(claims) -> tuple[str, ...]:
        return tuple(item["claim"] for item in claims if item["verdict"] != "supported")

    def judge_positive(
        self, case_id: str, question: str, answer: str, facts
    ) -> Verdict:
        raw = self.client.chat_json(
            model=self.model,
            system=POSITIVE_SYSTEM,
            prompt=(
                f"CORPUS:\n{self.corpus}\n\nPERGUNTA: {question}\n\n"
                f"RESPOSTA: {_clean(answer)}\n\nFATOS DE REFERÊNCIA: {list(facts)}"
            ),
            schema=POSITIVE_SCHEMA,
            strict=True,
        )
        complete = bool(raw["facts"]) and all(
            f["stated_correctly"] for f in raw["facts"]
        )
        false_claims = self._false_claims(raw["claims"])
        return Verdict(
            case_id,
            correct=complete and not false_claims and bool(raw["answers_question"]),
            complete=complete,
            false_claims=false_claims,
            detail=raw,
        )

    def judge_negative(
        self, case_id: str, question: str, answer: str, expected
    ) -> Verdict:
        raw = self.client.chat_json(
            model=self.model,
            system=NEGATIVE_SYSTEM,
            prompt=(
                f"CORPUS:\n{self.corpus}\n\nPERGUNTA: {question}\n\n"
                f"COMPORTAMENTO ESPERADO: {expected}\n\nRESPOSTA: {_clean(answer)}"
            ),
            schema=NEGATIVE_SCHEMA,
            strict=True,
        )
        fabricated = self._false_claims(raw["claims"])
        ok = bool(raw["declines_or_corrects"]) and not fabricated
        return Verdict(
            case_id, correct=ok, complete=ok, false_claims=fabricated, detail=raw
        )


def summarize(verdicts: list[Verdict]) -> dict[str, float | int]:
    total = len(verdicts)
    if not total:
        return {"total": 0}
    return {
        "total": total,
        "correct": sum(v.correct for v in verdicts),
        "accuracy": round(sum(v.correct for v in verdicts) / total, 4),
        "complete": sum(v.complete for v in verdicts),
        "with_false_claims": sum(bool(v.false_claims) for v in verdicts),
        "hallucination_rate": round(
            sum(bool(v.false_claims) for v in verdicts) / total, 4
        ),
    }


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _clean(answer: str) -> str:
    return re.sub(r"\s*\[[FG]\d+\]", "", answer).strip()
