"""Reavalia com o juiz as respostas já registradas em uma rodada, sem regenerar."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from trilhagraph.config import Settings
from trilhagraph.evaluation.evaluator import load_cases
from trilhagraph.evaluation.judge import Judge, summarize
from trilhagraph.ingestion.llm import OpenAIClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--modes", nargs="+", default=["hybrid", "vector"])
    args = parser.parse_args()
    settings = Settings.from_env()
    judge = Judge.from_directory(
        OpenAIClient.from_settings(settings),
        model=args.judge_model,
        directory=ROOT / "data/raw",
    )
    cases = {case.id: case for case in load_cases(args.dataset)}
    negatives = {n["id"]: n for n in json.loads(args.negatives.read_text())}
    for mode in args.modes:
        report = args.run / f"{mode}-report.json"
        if not report.exists():
            continue
        verdicts = []
        for item in json.loads(report.read_text())["cases"]:
            case = cases.get(item["id"])
            if case and not item.get("error"):
                verdicts.append(
                    judge.judge_positive(
                        case.id, case.question, item["answer"], case.expected_concepts
                    )
                )
        (args.run / f"{mode}-judge.json").write_text(
            json.dumps(
                {
                    "summary": summarize(verdicts),
                    "cases": [asdict(v) for v in verdicts],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        neg = []
        for nid, case in negatives.items():
            path = args.run / f"{mode}-{nid}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            if "result" in data:
                neg.append(
                    judge.judge_negative(
                        nid,
                        case["question"],
                        data["result"]["answer"],
                        case["expected_behavior"],
                    )
                )
        (args.run / f"{mode}-judge-negatives.json").write_text(
            json.dumps(
                {"summary": summarize(neg), "cases": [asdict(v) for v in neg]},
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
