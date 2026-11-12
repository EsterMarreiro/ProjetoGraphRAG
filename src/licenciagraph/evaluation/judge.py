"""Juiz LLM com acesso ao corpus completo, usado para medir acerto e alucinação."""

from __future__ import annotations

import re
from dataclasses import dataclass
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

CLAIM_RULES = """\
Decomponha a resposta em afirmações atômicas sobre licenciamento ambiental (ignore
marcadores [F1], avisos de caráter orientativo e frases de recusa). Para cada
afirmação, copie em quote a(s) frase(s) das EVIDÊNCIAS que a confirmam ou refutam;
use quote vazio se não houver. verdict: supported (as evidências confirmam),
contradicted (as evidências afirmam o contrário ou outro valor) ou unverifiable (as
evidências não tratam do assunto). As evidências são os trechos oficiais relevantes
para a pergunta, mas podem ser incompletas: afirmações plausíveis que não constam
delas são unverifiable, não contradicted. Afirmações que combinam fatos de várias
frases são supported se cada fato combinado é verdadeiro. Afirmar que a norma NÃO
informa algo é supported quando as evidências de fato não tratam do assunto."""

POSITIVE_SYSTEM = f"""Você é um revisor rigoroso. Recebe EVIDÊNCIAS oficiais (trechos
do Manual de Licenciamento Ambiental e da NA-101 da SUDEMA), uma pergunta, uma
resposta e fatos de referência.
{CLAIM_RULES}
Também diga, para cada fato de referência, se a resposta o afirma corretamente
(stated_correctly) e se a resposta atende ao que foi perguntado."""

NEGATIVE_SYSTEM = f"""Você é um revisor rigoroso. Recebe EVIDÊNCIAS oficiais (trechos
do Manual de Licenciamento Ambiental e da NA-101 da SUDEMA), uma pergunta cuja
resposta NÃO está nas normas (ou que traz premissa falsa) e o comportamento
esperado.
{CLAIM_RULES}
declines_or_corrects é verdadeiro se a resposta admite a ausência da informação
ou corrige a premissa falsa. Neste caso, qualquer afirmação factual que não conste
das evidências deve ser marcada como unverifiable."""


@dataclass(frozen=True, slots=True)
class Verdict:
    id: str
    correct: bool
    complete: bool
    false_claims: tuple[str, ...]
    detail: dict[str, Any]
    unsupported: tuple[str, ...] = ()


class Judge:
    def __init__(self, client: OpenAIClient, *, model: str) -> None:
        self.client = client
        self.model = model

    @staticmethod
    def _problems(claims, *, strict: bool) -> tuple[str, ...]:
        bad = {"contradicted"} if not strict else {"contradicted", "unverifiable"}
        return tuple(item["claim"] for item in claims if item["verdict"] in bad)

    def judge_positive(
        self, case_id: str, question: str, answer: str, facts, evidence
    ) -> Verdict:
        raw = self.client.chat_json(
            model=self.model,
            system=POSITIVE_SYSTEM,
            prompt=(
                f"EVIDÊNCIAS:\n{_evidence_text(evidence)}\n\nPERGUNTA: {question}\n\n"
                f"RESPOSTA: {_clean(answer)}\n\nFATOS DE REFERÊNCIA: {list(facts)}"
            ),
            schema=POSITIVE_SCHEMA,
            strict=True,
        )
        complete = bool(raw["facts"]) and all(
            f["stated_correctly"] for f in raw["facts"]
        )
        false_claims = self._problems(raw["claims"], strict=False)
        return Verdict(
            case_id,
            correct=complete and not false_claims and bool(raw["answers_question"]),
            complete=complete,
            false_claims=false_claims,
            detail=raw,
            unsupported=tuple(
                c["claim"] for c in raw["claims"] if c["verdict"] == "unverifiable"
            ),
        )

    def judge_negative(
        self, case_id: str, question: str, answer: str, expected, evidence=()
    ) -> Verdict:
        raw = self.client.chat_json(
            model=self.model,
            system=NEGATIVE_SYSTEM,
            prompt=(
                f"EVIDÊNCIAS:\n{_evidence_text(evidence)}\n\nPERGUNTA: {question}\n\n"
                f"COMPORTAMENTO ESPERADO: {expected}\n\nRESPOSTA: {_clean(answer)}"
            ),
            schema=NEGATIVE_SCHEMA,
            strict=True,
        )
        fabricated = self._problems(raw["claims"], strict=True)
        ok = bool(raw["declines_or_corrects"]) and not fabricated
        return Verdict(
            case_id, correct=ok, complete=ok, false_claims=fabricated, detail=raw
        )


def _evidence_text(evidence) -> str:
    if not evidence:
        return "(nenhuma evidência de referência para este caso)"
    return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(evidence, 1))


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
        "with_unsupported_claims": sum(bool(v.unsupported) for v in verdicts),
        "hallucination_rate": round(
            sum(bool(v.false_claims) for v in verdicts) / total, 4
        ),
    }


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _clean(answer: str) -> str:
    return re.sub(r"\s*\[[FG]\d+\]", "", answer).strip()
