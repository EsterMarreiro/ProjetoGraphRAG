"""Verificação qualitativa de dúvidas gerais: grava respostas para leitura manual."""

from __future__ import annotations

import json
from pathlib import Path

from licenciagraph.config import Settings
from licenciagraph.graph import GraphConnection
from licenciagraph.ingestion.llm import OpenAIClient
from licenciagraph.retrieval.generation import AnswerGenerator
from licenciagraph.retrieval.retriever import build_retriever
from licenciagraph.retrieval.service import QuestionAnsweringService

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = Settings.from_env()
    llm = OpenAIClient.from_settings(settings)
    cases = json.loads((ROOT / "data/evaluation/general_questions.json").read_text())
    rows = []
    with GraphConnection(settings) as connection:
        service = QuestionAnsweringService(
            retriever=build_retriever(connection.graph, llm, settings),
            generator=AnswerGenerator(
                llm, model=settings.chat_model, verifier_model=settings.verifier_model
            ),
        )
        for case in cases:
            answer = service.answer(case["question"])
            rows.append(
                {**case, "answer": answer.answer, "abstained": answer.abstained}
            )
    output = ROOT / "artifacts/general-check.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    for row in rows:
        print(row["id"], "ABST" if row["abstained"] else "ok", row["answer"][:160])


if __name__ == "__main__":
    main()
