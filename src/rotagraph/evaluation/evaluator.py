"""Metricas simples, transparentes e reproduziveis para o GraphRAG."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from rotagraph.retrieval.models import GraphEdge, GroundedAnswer


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    question: str
    expected_sources: tuple[str, ...]
    expected_concepts: tuple[str, ...]
    requires_multihop: bool = False


@dataclass(frozen=True, slots=True)
class CaseResult:
    id: str
    question: str
    answer: str
    retrieved_sources: tuple[str, ...]
    source_recall: float
    concept_coverage: float
    citation_validity: float
    has_graph_context: bool
    has_multihop_context: bool
    confidence: float
    passed: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    total: int
    completed: int
    passed: int
    pass_rate: float
    mean_source_recall: float
    mean_concept_coverage: float
    mean_citation_validity: float
    graph_context_rate: float
    multihop_context_rate: float
    mean_confidence: float
    cases: tuple[CaseResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnsweringService(Protocol):
    def answer(self, question: str) -> GroundedAnswer: ...


def load_cases(path: Path) -> list[EvaluationCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("O dataset de avaliacao deve conter uma lista JSON")
    cases: list[EvaluationCase] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Cada caso de avaliacao deve ser um objeto")
        cases.append(
            EvaluationCase(
                id=_required_text(item, "id"),
                question=_required_text(item, "question"),
                expected_sources=tuple(_required_list(item, "expected_sources")),
                expected_concepts=tuple(_required_list(item, "expected_concepts")),
                requires_multihop=bool(item.get("requires_multihop", False)),
            )
        )
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Os IDs dos casos de avaliacao devem ser unicos")
    return cases


def evaluate_service(
    service: AnsweringService, cases: list[EvaluationCase]
) -> EvaluationReport:
    results: list[CaseResult] = []
    for case in cases:
        try:
            answer = service.answer(case.question)
            results.append(evaluate_answer(case, answer))
        except Exception as exc:
            results.append(
                CaseResult(
                    id=case.id,
                    question=case.question,
                    answer="",
                    retrieved_sources=(),
                    source_recall=0.0,
                    concept_coverage=0.0,
                    citation_validity=0.0,
                    has_graph_context=False,
                    has_multihop_context=False,
                    confidence=0.0,
                    passed=False,
                    error=str(exc),
                )
            )
    return _summarize(results)


def evaluate_answer(case: EvaluationCase, answer: GroundedAnswer) -> CaseResult:
    retrieved = tuple(dict.fromkeys(hit.filename for hit in answer.retrieval.hits))
    expected = {_normalize_filename(source) for source in case.expected_sources}
    found = {_normalize_filename(source) for source in retrieved}
    source_recall = len(expected & found) / len(expected) if expected else 1.0

    normalized_answer = _normalize(answer.answer)
    covered = sum(
        _normalize(concept) in normalized_answer for concept in case.expected_concepts
    )
    concept_coverage = covered / len(case.expected_concepts)

    valid_citations = {f"F{index}" for index in range(1, len(retrieved) + 1)}
    citations = set(answer.citations)
    cited_in_text = set(re.findall(r"\[(F\d+)\]", answer.answer))
    citation_validity = float(
        bool(citations)
        and citations <= valid_citations
        and cited_in_text == citations
    )
    has_graph_context = bool(answer.retrieval.edges)
    has_multihop_context = _has_multihop_path(answer.retrieval.edges)
    multihop_ok = not case.requires_multihop or has_multihop_context
    passed = (
        source_recall == 1.0
        and concept_coverage >= 0.5
        and citation_validity == 1.0
        and multihop_ok
    )
    return CaseResult(
        id=case.id,
        question=case.question,
        answer=answer.answer,
        retrieved_sources=retrieved,
        source_recall=round(source_recall, 4),
        concept_coverage=round(concept_coverage, 4),
        citation_validity=citation_validity,
        has_graph_context=has_graph_context,
        has_multihop_context=has_multihop_context,
        confidence=round(answer.retrieval.confidence, 4),
        passed=passed,
    )


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        "# Relatorio de avaliacao GraphRAG",
        "",
        "## Resumo",
        "",
        "| Metrica | Resultado |",
        "|---|---:|",
        f"| Casos concluidos | {report.completed}/{report.total} |",
        f"| Taxa de aprovacao | {report.pass_rate:.1%} |",
        f"| Recall medio de fontes | {report.mean_source_recall:.1%} |",
        f"| Cobertura media de conceitos | {report.mean_concept_coverage:.1%} |",
        f"| Validade media de citacoes | {report.mean_citation_validity:.1%} |",
        f"| Casos com contexto de grafo | {report.graph_context_rate:.1%} |",
        f"| Casos com caminho multi-hop | {report.multihop_context_rate:.1%} |",
        f"| Confianca media | {report.mean_confidence:.3f} |",
        "",
        "## Casos",
        "",
        "| ID | Status | Fontes | Conceitos | Multi-hop | Confianca |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in report.cases:
        status = "OK" if result.passed else "FALHA"
        if result.error:
            status = "ERRO"
        lines.append(
            f"| {result.id} | {status} | {result.source_recall:.0%} | "
            f"{result.concept_coverage:.0%} | "
            f"{'sim' if result.has_multihop_context else 'nao'} | "
            f"{result.confidence:.3f} |"
        )
        if result.error:
            lines.extend(["", f"> {result.id}: {result.error}"])
    lines.append("")
    return "\n".join(lines)


def _summarize(results: list[CaseResult]) -> EvaluationReport:
    total = len(results)
    completed = sum(result.error is None for result in results)
    passed = sum(result.passed for result in results)
    return EvaluationReport(
        total=total,
        completed=completed,
        passed=passed,
        pass_rate=_mean([float(result.passed) for result in results]),
        mean_source_recall=_mean([result.source_recall for result in results]),
        mean_concept_coverage=_mean([result.concept_coverage for result in results]),
        mean_citation_validity=_mean([result.citation_validity for result in results]),
        graph_context_rate=_mean(
            [float(result.has_graph_context) for result in results]
        ),
        multihop_context_rate=_mean(
            [float(result.has_multihop_context) for result in results]
        ),
        mean_confidence=_mean([result.confidence for result in results]),
        cases=tuple(results),
    )


def _mean(values: list[float]) -> float:
    return round(fmean(values), 4) if values else 0.0


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value).strip()


def _normalize_filename(value: str) -> str:
    return Path(value).name.casefold()


def _has_multihop_path(edges: list[GraphEdge]) -> bool:
    """Detecta duas arestas conectadas, ignorando a direcao da travessia."""
    for index, first in enumerate(edges):
        first_nodes = {first.source, first.target}
        for second in edges[index + 1 :]:
            second_nodes = {second.source, second.target}
            if first_nodes & second_nodes and first_nodes != second_nodes:
                return True
    return False


def _required_text(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Campo obrigatorio invalido: {key}")
    return value.strip()


def _required_list(item: dict[str, Any], key: str) -> list[str]:
    value = item.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(entry, str) and entry.strip() for entry in value
    ):
        raise ValueError(f"Campo obrigatorio invalido: {key}")
    return [entry.strip() for entry in value]
