"""Preferências de aprendizagem explicitadas na consulta."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LearningFilters:
    area: str = ""
    competency: str = ""
    profile: str = ""
    level: str = ""

    def augment_question(self, question: str) -> str:
        criteria = [
            f"{label}: {value.strip()}"
            for label, value in (
                ("área", self.area),
                ("competência", self.competency),
                ("perfil", self.profile),
                ("nível", self.level),
            )
            if value.strip()
        ]
        if not criteria:
            return question.strip()
        return f"{question.strip()}\n\nPreferências explícitas: {'; '.join(criteria)}."
