"""Avaliacao reproduzivel do sistema GraphRAG."""

from licenciagraph.evaluation.evaluator import (
    EvaluationCase,
    EvaluationReport,
    evaluate_service,
    load_cases,
    render_markdown,
)

__all__ = [
    "EvaluationCase",
    "EvaluationReport",
    "evaluate_service",
    "load_cases",
    "render_markdown",
]
