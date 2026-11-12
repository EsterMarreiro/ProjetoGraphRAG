"""Experimento com LLM real, respostas completas e checkpoints por pergunta."""

from __future__ import annotations

import argparse
import hashlib
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

from licenciagraph.config import Settings
from licenciagraph.evaluation.evaluator import (
    _summarize,
    evaluate_service,
    load_cases,
    render_markdown,
)
from licenciagraph.evaluation.judge import Judge, summarize
from licenciagraph.graph import GraphConnection, SchemaManager
from licenciagraph.ingestion.curated import build_documents
from licenciagraph.ingestion.facts import FactsExtractor
from licenciagraph.ingestion.llm import OpenAIClient
from licenciagraph.ingestion.pipeline import IngestionPipeline
from licenciagraph.ingestion.writer import GraphWriter
from licenciagraph.retrieval.generation import AnswerGenerator
from licenciagraph.retrieval.retriever import build_retriever
from licenciagraph.retrieval.service import QuestionAnsweringService

ROOT = Path(__file__).resolve().parents[1]
PRICES_USD_PER_MILLION = {
    "gpt-4.1": (2.0, 8.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "text-embedding-3-small": (0.02, 0.0),
}


def cost_usd(*clients) -> tuple[float, dict]:
    totals: dict[str, dict[str, int]] = {}
    for client in clients:
        for model, usage in client.usage.items():
            entry = totals.setdefault(model, {"input": 0, "output": 0, "calls": 0})
            for key in entry:
                entry[key] += usage[key]
    total = 0.0
    for model, usage in totals.items():
        price_in, price_out = PRICES_USD_PER_MILLION.get(model, (0.0, 0.0))
        usage["usd"] = round(
            (usage["input"] * price_in + usage["output"] * price_out) / 1e6, 4
        )
        total += usage["usd"]
    return round(total, 4), totals


def write_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    temporary.replace(path)


class AuditedLLM(OpenAIClient):
    def __init__(self, settings, directory):
        super().__init__(
            settings.openai_api_key or "",
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout,
        )
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
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--skip-negatives", action="store_true")
    parser.add_argument("--skip-positives", action="store_true")
    parser.add_argument(
        "--modes", nargs="+", choices=["hybrid", "vector"], default=["hybrid", "vector"]
    )
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/evaluation/questions.json"
    )
    parser.add_argument(
        "--negatives",
        type=Path,
        default=ROOT / "data/evaluation/negative_questions.json",
    )
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--max-usd", type=float, default=0.0)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit deve ser positivo")
    cases = load_cases(args.dataset)
    raw_cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.case_ids:
        unknown = set(args.case_ids) - {case.id for case in cases}
        if unknown:
            parser.error(f"Casos desconhecidos: {sorted(unknown)}")
        cases = [case for case in cases if case.id in args.case_ids]
    if args.limit:
        cases = cases[: args.limit]
    if args.skip_positives:
        cases = []
    directory = args.output or ROOT / "artifacts" / datetime.now(UTC).strftime(
        "run-%Y%m%dT%H%M%SZ"
    )
    directory.mkdir(parents=True, exist_ok=False)
    capture(directory)
    settings = Settings.from_env()
    llm = AuditedLLM(settings, directory)
    judge_client = OpenAIClient.from_settings(settings)
    judge = None if args.no_judge else Judge(judge_client, model=args.judge_model)

    def check_budget() -> None:
        spent, usage = cost_usd(llm, judge_client)
        write_json(directory / "usage.json", {"usd": spent, "models": usage})
        if args.max_usd and spent > args.max_usd:
            raise SystemExit(f"Orçamento excedido: US$ {spent:.2f} > {args.max_usd}")

    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_diff": subprocess.check_output(["git", "diff", "--stat"], text=True),
        "settings": {
            k: v
            for k, v in asdict(settings).items()
            if "password" not in k and "api_key" not in k
        },
        "packages": {
            name: version(name) for name in ["falkordb", "streamlit", "licenciagraph"]
        },
        "models": {"chat": settings.chat_model, "embedding": settings.embedding_model},
        "chat_options": llm.chat_options,
        "corpus": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "data/curated").glob("*.json"))
        },
        "modes": args.modes,
        "case_ids": [case.id for case in cases],
        "include_negatives": not args.skip_negatives,
        "dataset": str(args.dataset.name),
        "tag": args.tag,
        "judge_model": None if args.no_judge else args.judge_model,
    }
    write_json(directory / "metadata.json", metadata)
    print(f"Experimento: {directory}", flush=True)
    with GraphConnection(settings) as connection:
        SchemaManager(
            connection.graph, embedding_dimension=settings.embedding_dimension
        ).ensure()
        if args.ingest:
            report = IngestionPipeline(
                writer=GraphWriter(connection.graph),
                extractor=FactsExtractor(llm, model=settings.extraction_model),
                llm=llm,
                embedding_model=settings.embedding_model,
                embedding_dimension=settings.embedding_dimension,
            ).run(build_documents(ROOT / "data/curated"), force=args.force_ingest)
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
        for mode in args.modes:
            service = RecordedService(
                QuestionAnsweringService(
                    retriever=build_retriever(connection.graph, llm, settings, mode),
                    generator=AnswerGenerator(
                        llm,
                        model=settings.chat_model,
                        verifier_model=settings.verifier_model,
                    ),
                ),
                directory / f"{mode}-answers.jsonl",
            )
            results = []
            verdicts = []
            evidence = {item["id"]: item["evidence"] for item in raw_cases}
            facts = {item["id"]: item["expected_facts"] for item in raw_cases}
            for case in cases:
                print(f"{mode}: {case.id}", flush=True)
                check_budget()
                results.extend(evaluate_service(service, [case]).cases)
                summary = _summarize(results)
                write_json(directory / f"{mode}-report.json", summary.as_dict())
                (directory / f"{mode}-report.md").write_text(render_markdown(summary))
                if judge:
                    last = results[-1]
                    if not last.error:
                        verdicts.append(
                            judge.judge_positive(
                                case.id,
                                case.question,
                                last.answer,
                                facts[case.id],
                                evidence[case.id],
                            )
                        )
                    write_json(
                        directory / f"{mode}-judge.json",
                        {
                            "summary": summarize(verdicts),
                            "cases": [asdict(v) for v in verdicts],
                        },
                    )
            negatives = (
                [] if args.skip_negatives else json.loads(args.negatives.read_text())
            )
            neg_verdicts = []
            for case in negatives:
                print(f"{mode}: {case['id']}", flush=True)
                check_budget()
                try:
                    answer = service.answer(case["question"])
                    entry = {
                        **case,
                        "result": asdict(answer),
                        "review_status": "pending_human_review",
                    }
                    if judge:
                        verdict = judge.judge_negative(
                            case["id"],
                            case["question"],
                            answer.answer,
                            case["expected_behavior"],
                            case.get("evidence", ()),
                        )
                        neg_verdicts.append(verdict)
                        entry["judge"] = asdict(verdict)
                    write_json(directory / f"{mode}-{case['id']}.json", entry)
                except Exception as exc:
                    write_json(
                        directory / f"{mode}-{case['id']}.json",
                        {**case, "error": str(exc)},
                    )
            if judge and neg_verdicts:
                write_json(
                    directory / f"{mode}-judge-negatives.json",
                    {
                        "summary": summarize(neg_verdicts),
                        "cases": [asdict(v) for v in neg_verdicts],
                    },
                )
    check_budget()
    metadata["usd"] = cost_usd(llm, judge_client)[0]
    metadata["completed_at"] = datetime.now(UTC).isoformat()
    write_json(directory / "metadata.json", metadata)
    print("Experimento concluído", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
