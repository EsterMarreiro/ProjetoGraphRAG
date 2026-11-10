"""Smoke test manual da persistencia no FalkorDB.

Executar com: python tests/integration_smoke.py
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from rotagraph.config import Settings
from rotagraph.graph import GraphConnection, SchemaManager
from rotagraph.ingestion.models import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from rotagraph.ingestion.pipeline import IngestionPipeline
from rotagraph.ingestion.writer import GraphWriter
from rotagraph.retrieval.algorithms import GraphAlgorithms
from rotagraph.retrieval.repository import RetrievalRepository


class SmokeExtractor:
    def extract(self, text: str) -> ExtractionResult:
        return ExtractionResult(
            entities=[
                ExtractedEntity(
                    "atracao",
                    "Atracao",
                    "Centro Histórico de Teste",
                    {"tipo": "historica", "municipio": "Cidade Teste", "uf": "TT"},
                ),
                ExtractedEntity(
                    "localidade",
                    "Localidade",
                    "Cidade Teste",
                    {"tipo": "municipio", "uf": "TT"},
                ),
            ],
            relationships=[
                ExtractedRelationship("atracao", "localidade", "LOCALIZADA_EM", 1.0)
            ],
        )


class SmokeEmbeddings:
    def embed(self, *, model: str, texts: list[str], dimensions: int):
        return [[0.01] * dimensions for _ in texts]


def main() -> None:
    settings = replace(Settings.from_env(), falkordb_graph="rotagraph_smoke_test")
    with GraphConnection(settings) as connection:
        try:
            SchemaManager(
                connection.graph,
                embedding_dimension=settings.embedding_dimension,
            ).ensure()
            with tempfile.TemporaryDirectory() as directory:
                data_dir = Path(directory)
                (data_dir / "smoke.txt").write_text(
                    "Cidade Teste\n\nO Centro Histórico de Teste está localizado em "
                    "Cidade Teste, no estado TT.",
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
                "MATCH ()-[r:LOCALIZADA_EM]->() RETURN count(r)"
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
            text_hits = repository.fulltext_search("Cidade Teste", limit=3)
            nodes, edges = repository.expand([vector_hits[0].chunk_id])
            algorithms = GraphAlgorithms(connection.graph)
            path = algorithms.shortest_path(
                "atracao:centro-historico-de-teste-cidade-teste-tt",
                "localidade:cidade-teste-tt",
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
            print(
                "Smoke test concluido: ingestao, busca vetorial/full-text, "
                "expansao multi-hop, shortest path e PageRank."
            )
        finally:
            connection.graph.delete()


if __name__ == "__main__":
    main()
