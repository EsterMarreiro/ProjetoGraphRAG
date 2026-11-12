"""Preferências explicitadas pelo empreendedor e acrescentadas à pergunta."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProcessFilters:
    activity: str = ""
    modality: str = ""
    porte: str = ""

    def augment_question(self, question: str) -> str:
        criteria = [
            f"{label}: {value.strip()}"
            for label, value in (
                ("atividade", self.activity),
                ("modalidade", self.modality),
                ("porte", self.porte),
            )
            if value.strip()
        ]
        if not criteria:
            return question.strip()
        return f"{question.strip()}\n\nPreferências explícitas: {'; '.join(criteria)}."
