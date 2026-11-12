"""Caso de uso completo: recuperar e gerar resposta."""

from __future__ import annotations

from licenciagraph.retrieval.generation import AnswerGenerator
from licenciagraph.retrieval.models import GroundedAnswer
from licenciagraph.retrieval.retriever import HybridRetriever


class QuestionAnsweringService:
    def __init__(
        self, *, retriever: HybridRetriever, generator: AnswerGenerator
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str) -> GroundedAnswer:
        return self.generator.generate(self.retriever.retrieve(question))
