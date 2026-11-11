"""Smoke test manual da persistencia no FalkorDB.

Executar com: python tests/integration_smoke.py
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

from trilhagraph.config import Settings
from trilhagraph.graph import GraphConnection, SchemaManager
from trilhagraph.ingestion.loader import load_documents
from trilhagraph.ingestion.models import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from trilhagraph.ingestion.pipeline import IngestionPipeline
from trilhagraph.ingestion.writer import GraphWriter
from trilhagraph.retrieval.algorithms import GraphAlgorithms
from trilhagraph.retrieval.planning import CYPHER_CHOICES, CypherPlan
from trilhagraph.retrieval.repository import RetrievalRepository


class SmokeExtractor:
    def extract(self, text: str) -> ExtractionResult:
        return ExtractionResult(
            entities=[
                ExtractedEntity(
                    "atracao",
                    "Curso",
                    "Curso Teste",
                    {"tipo": "didatico"},
                ),
                ExtractedEntity(
                    "localidade",
                    "Area",
                    "Dados",
                    {"tipo": "conhecimento"},
                ),
            ],
            relationships=[
                ExtractedRelationship("atracao", "localidade", "ABORDA", 1.0)
            ],
        )


class SmokeEmbeddings:
    def embed(self, *, model: str, texts: list[str], dimensions: int):
        return [[0.01] * dimensions for _ in texts]


def main() -> None:
    settings = replace(
        Settings.from_env(), falkordb_graph=f"trilhagraph_smoke_{uuid.uuid4().hex}"
    )
    with GraphConnection(settings) as connection:
        try:
            SchemaManager(
                connection.graph,
                embedding_dimension=settings.embedding_dimension,
            ).ensure()
            with tempfile.TemporaryDirectory() as directory:
                data_dir = Path(directory)
                (data_dir / "smoke.txt").write_text(
                    "Dados\n\nO Curso Teste aborda a área Dados.",
                    encoding="utf-8",
                )
                report = IngestionPipeline(
                    writer=GraphWriter(connection.graph),
                    extractor=SmokeExtractor(),
                    ollama=SmokeEmbeddings(),
                    embedding_model="fake",
                    embedding_dimension=settings.embedding_dimension,
                ).run(data_dir)

            count = connection.graph.ro_query("MATCH (n) RETURN count(n)").result_set
            relations = connection.graph.ro_query(
                "MATCH ()-[r:ABORDA]->() RETURN count(r)"
            ).result_set
            if (
                report.documents_processed != 1
                or count[0][0] != 4
                or relations[0][0] != 1
            ):
                raise AssertionError(
                    "Smoke test inesperado: "
                    f"report={report}, nodes={count}, relations={relations}"
                )
            repository = RetrievalRepository(connection.graph)
            vector_hits = repository.vector_search(
                [0.01] * settings.embedding_dimension, limit=3
            )
            text_hits = repository.fulltext_search("Dados", limit=3)
            nodes, edges = repository.expand([vector_hits[0].chunk_id])
            algorithms = GraphAlgorithms(connection.graph)
            path = algorithms.shortest_path(
                "curso:curso-teste",
                "area:dados",
            )
            ranking = algorithms.pagerank(limit=3)
            if (
                not vector_hits
                or not text_hits
                or len(nodes) != 2
                or len(edges) != 1
                or path is None
                or path.length != 1
                or not ranking
            ):
                raise AssertionError(
                    "Smoke test de recuperacao retornou dados incompletos"
                )
            query = next(
                query
                for query in CYPHER_CHOICES
                if "a:Curso" in query and ":ABORDA*" in query
            )
            structured = repository.structured_search(CypherPlan(query, "Curso Teste"))
            if not structured or structured[0].chunk_id != vector_hits[0].chunk_id:
                raise AssertionError("Cypher gerado não recuperou a fonte esperada")

            with tempfile.TemporaryDirectory() as directory:
                data_dir = Path(directory)
                for name in ("source_a.txt", "source_b.txt"):
                    (data_dir / name).write_text("Dados\n\nMesmo fato.")
                writer = GraphWriter(connection.graph)
                pipeline = IngestionPipeline(
                    writer=writer,
                    extractor=SmokeExtractor(),
                    ollama=SmokeEmbeddings(),
                    embedding_model="fake",
                    embedding_dimension=settings.embedding_dimension,
                )
                pipeline.run(data_dir)
                documents = load_documents(data_dir)
                chunk_ids = [f"{document.id}:trecho:0001" for document in documents]
                _, evidence_edges = repository.expand(chunk_ids)
                if len(evidence_edges) != 3:
                    raise AssertionError("Fontes do mesmo fato foram sobrescritas")
                writer.prepare_document(documents[0])
                if writer.is_document_current(documents[0]):
                    raise AssertionError("Carga incompleta foi marcada como atual")
                count = connection.graph.ro_query(
                    "MATCH ()-[r:ABORDA]->() RETURN count(r)"
                ).result_set[0][0]
                if count != 2:
                    raise AssertionError("Recarga não removeu só a evidência antiga")
                resumed = pipeline.run(data_dir)
                if resumed.documents_processed != 1 or resumed.documents_skipped != 1:
                    raise AssertionError("Retomada de carga não foi idempotente")
            print(
                "Smoke test concluido: ingestao, busca vetorial/full-text, "
                "expansao multi-hop, shortest path e PageRank."
            )
        finally:
            connection.graph.delete()


if __name__ == "__main__":
    main()
