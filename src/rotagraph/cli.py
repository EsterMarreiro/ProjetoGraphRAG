"""Comandos operacionais do projeto."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rotagraph.config import Settings
from rotagraph.evaluation import evaluate_service, load_cases, render_markdown
from rotagraph.graph import GraphConnection, SchemaManager
from rotagraph.ingestion.extraction import TourismExtractor
from rotagraph.ingestion.ollama import OllamaClient
from rotagraph.ingestion.pipeline import IngestionPipeline
from rotagraph.ingestion.writer import GraphWriter
from rotagraph.retrieval.generation import AnswerGenerator
from rotagraph.retrieval.repository import RetrievalRepository
from rotagraph.retrieval.retriever import HybridRetriever
from rotagraph.retrieval.service import QuestionAnsweringService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rotagraph")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("health", help="verifica a conexao com o FalkorDB")
    commands.add_parser("setup-schema", help="cria indices e constraints")
    ingest = commands.add_parser("ingest", help="ingere os arquivos TXT")
    ingest.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ingest.add_argument("--force", action="store_true")
    ask = commands.add_parser("ask", help="responde usando recuperacao hibrida")
    ask.add_argument("question", help="pergunta em linguagem natural")
    evaluate = commands.add_parser(
        "evaluate", help="executa o conjunto de avaliacao ponta a ponta"
    )
    evaluate.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/questions.json"),
    )
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--json-output", type=Path)
    evaluate.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        with GraphConnection(settings) as connection:
            if args.command == "health":
                payload = connection.healthcheck()
            elif args.command == "setup-schema":
                connection.healthcheck()
                payload = (
                    SchemaManager(
                        connection.graph,
                        embedding_dimension=settings.embedding_dimension,
                    )
                    .ensure()
                    .as_dict()
                )
            elif args.command == "ingest":
                connection.healthcheck()
                SchemaManager(
                    connection.graph,
                    embedding_dimension=settings.embedding_dimension,
                ).ensure()
                ollama = OllamaClient(
                    settings.ollama_base_url, timeout=settings.ollama_timeout
                )
                payload = (
                    IngestionPipeline(
                        writer=GraphWriter(connection.graph),
                        extractor=TourismExtractor(ollama, model=settings.chat_model),
                        ollama=ollama,
                        embedding_model=settings.embedding_model,
                        embedding_dimension=settings.embedding_dimension,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                    )
                    .run(args.data_dir, force=args.force)
                    .as_dict()
                )
            elif args.command == "ask":
                connection.healthcheck()
                service = _build_answering_service(settings, connection)
                answer = service.answer(args.question)
                payload = {
                    "answer": answer.answer,
                    "citations": answer.citations,
                    "caveats": answer.caveats,
                    "confidence": answer.retrieval.confidence,
                    "sources": [
                        {
                            "id": f"F{index}",
                            "file": hit.filename,
                            "title": hit.title,
                            "score": hit.score,
                            "channels": sorted(hit.channels),
                        }
                        for index, hit in enumerate(answer.retrieval.hits, start=1)
                    ],
                    "graph": {
                        "nodes": [
                            {"id": node.id, "label": node.label, "name": node.name}
                            for node in answer.retrieval.nodes
                        ],
                        "edges": [
                            {
                                "source": edge.source,
                                "target": edge.target,
                                "type": edge.relationship_type,
                                "chunk_id": edge.chunk_id,
                            }
                            for edge in answer.retrieval.edges
                        ],
                    },
                    "queries": answer.retrieval.queries,
                }
            else:
                connection.healthcheck()
                cases = load_cases(args.dataset)
                if args.limit is not None:
                    if args.limit <= 0:
                        raise ValueError("--limit deve ser maior que zero")
                    cases = cases[: args.limit]
                report = evaluate_service(
                    _build_answering_service(settings, connection), cases
                )
                payload = report.as_dict()
                if args.json_output:
                    _write_text(
                        args.json_output,
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    )
                if args.markdown_output:
                    _write_text(args.markdown_output, render_markdown(report))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


def _build_answering_service(
    settings: Settings, connection: GraphConnection
) -> QuestionAnsweringService:
    ollama = OllamaClient(settings.ollama_base_url, timeout=settings.ollama_timeout)
    return QuestionAnsweringService(
        retriever=HybridRetriever(
            repository=RetrievalRepository(connection.graph),
            ollama=ollama,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
        ),
        generator=AnswerGenerator(ollama, model=settings.chat_model),
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
