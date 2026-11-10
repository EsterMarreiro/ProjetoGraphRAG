"""Avaliacao reproduzivel do sistema GraphRAG."""

from rotagraph.evaluation.evaluator import (
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
