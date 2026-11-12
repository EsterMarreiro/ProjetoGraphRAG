"""Montagem das dependencias usadas pela interface."""

from __future__ import annotations

from dataclasses import dataclass

from licenciagraph.config import Settings
from licenciagraph.graph import GraphConnection
from licenciagraph.guide import LicensingGuide
from licenciagraph.ingestion.llm import OpenAIClient
from licenciagraph.retrieval.generation import AnswerGenerator
from licenciagraph.retrieval.planning import CypherPlanner
from licenciagraph.retrieval.repository import RetrievalRepository
from licenciagraph.retrieval.retriever import HybridRetriever
from licenciagraph.retrieval.service import QuestionAnsweringService


@dataclass(slots=True)
class UIRuntime:
    settings: Settings
    connection: GraphConnection
    llm: OpenAIClient
    service: QuestionAnsweringService
    guide: LicensingGuide

    def close(self) -> None:
        self.connection.close()


def build_runtime(settings: Settings | None = None) -> UIRuntime:
    settings = settings or Settings.from_env()
    connection = GraphConnection(settings)
    graph = connection.connect()
    llm = OpenAIClient.from_settings(settings)
    service = QuestionAnsweringService(
        retriever=HybridRetriever(
            repository=RetrievalRepository(graph),
            llm=llm,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
            planner=CypherPlanner(llm, model=settings.chat_model),
        ),
        generator=AnswerGenerator(
            llm,
            model=settings.chat_model,
            verifier_model=settings.verifier_model,
        ),
    )
    return UIRuntime(settings, connection, llm, service, LicensingGuide(graph))
