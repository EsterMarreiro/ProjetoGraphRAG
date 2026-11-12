"""Contrato comum dos extratores usados pelo pipeline de ingestão."""

from __future__ import annotations

from typing import Protocol

from licenciagraph.ingestion.models import ExtractionResult


class Extractor(Protocol):
    def extract(self, text: str) -> ExtractionResult: ...
