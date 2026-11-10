"""Montagem das dependencias usadas pela interface."""

from __future__ import annotations

from dataclasses import dataclass

from rotagraph.config import Settings
from rotagraph.graph import GraphConnection
from rotagraph.ingestion.ollama import OllamaClient
from rotagraph.retrieval.generation import AnswerGenerator
from rotagraph.retrieval.repository import RetrievalRepository
from rotagraph.retrieval.retriever import HybridRetriever
from rotagraph.retrieval.service import QuestionAnsweringService


@dataclass(slots=True)
class UIRuntime:
    settings: Settings
    connection: GraphConnection
    ollama: OllamaClient
    service: QuestionAnsweringService

    def close(self) -> None:
        self.connection.close()


def build_runtime(settings: Settings | None = None) -> UIRuntime:
    settings = settings or Settings.from_env()
    connection = GraphConnection(settings)
    graph = connection.connect()
    ollama = OllamaClient(settings.ollama_base_url, timeout=settings.ollama_timeout)
    service = QuestionAnsweringService(
        retriever=HybridRetriever(
            repository=RetrievalRepository(graph),
            ollama=ollama,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
        ),
        generator=AnswerGenerator(ollama, model=settings.chat_model),
    )
    return UIRuntime(settings, connection, ollama, service)
