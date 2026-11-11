"""Experimento com LLM real, respostas completas e checkpoints por pergunta."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from capture_sources import capture

from trilhagraph.config import Settings
from trilhagraph.evaluation.evaluator import (
    _summarize,
    evaluate_service,
    load_cases,
    render_markdown,
)
from trilhagraph.graph import GraphConnection, SchemaManager
from trilhagraph.ingestion.extraction import LearningExtractor
from trilhagraph.ingestion.loader import load_documents
from trilhagraph.ingestion.ollama import OllamaClient
from trilhagraph.ingestion.pipeline import IngestionPipeline
from trilhagraph.ingestion.writer import GraphWriter
from trilhagraph.retrieval.generation import AnswerGenerator
from trilhagraph.retrieval.planning import CypherPlanner
from trilhagraph.retrieval.repository import RetrievalRepository
from trilhagraph.retrieval.retriever import HybridRetriever
from trilhagraph.retrieval.service import QuestionAnsweringService

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    temporary.replace(path)


class AuditedOllama(OllamaClient):
    def __init__(self, settings, directory):
        super().__init__(settings.ollama_base_url, timeout=settings.ollama_timeout)
        self.directory = directory
        self.calls = 0

    def chat_json(self, **kwargs):
        self.calls += 1
        destination = self.directory / f"llm-{self.calls:04}.json"
        record = {"started_at": datetime.now(UTC).isoformat(), "request": kwargs}
        started = time.monotonic()
        print(f"LLM chamada {self.calls}: {kwargs['model']}", flush=True)
        write_json(destination, record)
        try:
            result = super().chat_json(**kwargs)
            record["response"] = result
            return result
        except Exception as exc:
            record["error"] = str(exc)
            raise
        finally:
            record["seconds"] = round(time.monotonic() - started, 3)
            write_json(destination, record)


class RecordedService:
    def __init__(self, service, path):
        self.service, self.path = service, path

    def answer(self, question):
        started = time.monotonic()
        answer = self.service.answer(question)
        with self.path.open("a") as output:
            output.write(
                json.dumps(
                    {"seconds": time.monotonic() - started, "result": asdict(answer)},
                    ensure_ascii=False,
                    default=sorted,
                )
                + "\n"
            )
        return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--modes", nargs="+", choices=["hybrid", "vector"], default=["hybrid", "vector"]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or ROOT / "artifacts" / datetime.now(UTC).strftime(
        "run-%Y%m%dT%H%M%SZ"
    )
    directory.mkdir(parents=True, exist_ok=False)
    capture(directory)
    settings = Settings.from_env()
    ollama = AuditedOllama(settings, directory)
    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_diff": subprocess.check_output(["git", "diff", "--stat"], text=True),
        "settings": {k: v for k, v in asdict(settings).items() if "password" not in k},
        "packages": {
            name: version(name) for name in ["falkordb", "streamlit", "trilhagraph"]
        },
        "models": ollama._get("/api/tags"),
        "chat_options": ollama.chat_options,
        "corpus": {
            doc.path.name: doc.checksum for doc in load_documents(ROOT / "data/raw")
        },
        "modes": args.modes,
    }
    write_json(directory / "metadata.json", metadata)
    print(f"Experimento: {directory}", flush=True)
    with GraphConnection(settings) as connection:
        SchemaManager(
            connection.graph, embedding_dimension=settings.embedding_dimension
        ).ensure()
        if not args.skip_ingest:
            report = IngestionPipeline(
                writer=GraphWriter(connection.graph),
                extractor=LearningExtractor(ollama, model=settings.chat_model),
                ollama=ollama,
                embedding_model=settings.embedding_model,
                embedding_dimension=settings.embedding_dimension,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            ).run(ROOT / "data/raw", force=args.force_ingest)
            write_json(directory / "ingestion.json", report.as_dict())
        write_json(
            directory / "graph-stats.json",
            {
                "labels": connection.graph.ro_query(
                    "MATCH (n) RETURN labels(n)[0],count(n)"
                ).result_set,
                "relationships": connection.graph.ro_query(
                    "MATCH ()-[r]->() RETURN type(r),count(r)"
                ).result_set,
            },
        )
        cases = load_cases(ROOT / "data/evaluation/questions.json")
        if args.limit:
            cases = cases[: args.limit]
        for mode in args.modes:
            service = RecordedService(
                QuestionAnsweringService(
                    retriever=HybridRetriever(
                        repository=RetrievalRepository(connection.graph),
                        ollama=ollama,
                        embedding_model=settings.embedding_model,
                        embedding_dimension=settings.embedding_dimension,
                        planner=CypherPlanner(ollama, model=settings.chat_model),
                        mode=mode,
                    ),
                    generator=AnswerGenerator(ollama, model=settings.chat_model),
                ),
                directory / f"{mode}-answers.jsonl",
            )
            results = []
            for case in cases:
                print(f"{mode}: {case.id}", flush=True)
                results.extend(evaluate_service(service, [case]).cases)
                summary = _summarize(results)
                write_json(directory / f"{mode}-report.json", summary.as_dict())
                (directory / f"{mode}-report.md").write_text(render_markdown(summary))
            negatives = json.loads(
                (ROOT / "data/evaluation/negative_questions.json").read_text()
            )
            for case in negatives:
                print(f"{mode}: {case['id']}", flush=True)
                try:
                    answer = service.answer(case["question"])
                    write_json(
                        directory / f"{mode}-{case['id']}.json",
                        {
                            **case,
                            "result": asdict(answer),
                            "review_status": "pending_human_review",
                        },
                    )
                except Exception as exc:
                    write_json(
                        directory / f"{mode}-{case['id']}.json",
                        {**case, "error": str(exc)},
                    )
    metadata["completed_at"] = datetime.now(UTC).isoformat()
    write_json(directory / "metadata.json", metadata)
    print("Experimento concluído", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
