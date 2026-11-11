"""Caso de uso completo: recuperar e gerar resposta."""

from __future__ import annotations

from trilhagraph.retrieval.generation import AnswerGenerator
from trilhagraph.retrieval.models import GroundedAnswer
from trilhagraph.retrieval.retriever import HybridRetriever


class QuestionAnsweringService:
    def __init__(
        self, *, retriever: HybridRetriever, generator: AnswerGenerator
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str) -> GroundedAnswer:
        return self.generator.generate(self.retriever.retrieve(question))
