"""Recalcula métricas sobre as mesmas respostas salvas, sem nova inferência."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from trilhagraph.evaluation.evaluator import (
    CaseResult,
    _summarize,
    evaluate_answer,
    load_cases,
    render_markdown,
)
from trilhagraph.retrieval.models import (
    GraphEdge,
    GraphNode,
    GroundedAnswer,
    RetrievalResult,
    SearchHit,
)

ROOT = Path(__file__).resolve().parents[1]


def restore(raw):
    retrieval = raw["retrieval"]
    return GroundedAnswer(
        answer=raw["answer"],
        citations=raw["citations"],
        caveats=raw["caveats"],
        supporting_edges=raw.get("supporting_edges", []),
        abstained=raw.get("abstained", False),
        retrieval=RetrievalResult(
            question=retrieval["question"],
            hits=[
                SearchHit(**{**hit, "channels": set(hit["channels"])})
                for hit in retrieval["hits"]
            ],
            nodes=[GraphNode(**node) for node in retrieval["nodes"]],
            edges=[GraphEdge(**edge) for edge in retrieval["edges"]],
            queries=retrieval["queries"],
            confidence=retrieval["confidence"],
            warnings=retrieval.get("warnings", []),
        ),
    )


def main():
    directory = Path(sys.argv[1])
    cases = {
        case.id: case for case in load_cases(ROOT / "data/evaluation/questions.json")
    }
    for mode in ("hybrid", "vector"):
        report_path = directory / f"{mode}-report.json"
        answer_path = directory / f"{mode}-answers.jsonl"
        if not report_path.exists() or not answer_path.exists():
            continue
        old = json.loads(report_path.read_text())
        answers = {}
        for line in answer_path.read_text().splitlines():
            answer = restore(json.loads(line)["result"])
            answers[answer.retrieval.question] = answer
        results = []
        for result in old["cases"]:
            case = cases[result["id"]]
            if case.question in answers:
                results.append(evaluate_answer(case, answers[case.question]))
            else:
                results.append(CaseResult(**result))
        report = _summarize(results)
        report_path.write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
        )
        (directory / f"{mode}-report.md").write_text(render_markdown(report))
        print(mode, report.content_check_rate, report.pass_rate)


if __name__ == "__main__":
    main()
