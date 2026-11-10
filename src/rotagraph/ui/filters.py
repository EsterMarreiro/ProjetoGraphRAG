"""Filtros turisticos aplicados de forma transparente a pergunta."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TourismFilters:
    regions: tuple[str, ...] = ()
    state: str = ""
    activity: str = ""
    profile: str = ""
    period: str = ""

    def augment_question(self, question: str) -> str:
        criteria: list[str] = []
        if self.regions:
            criteria.append(f"regiões: {', '.join(self.regions)}")
        if self.state.strip():
            criteria.append(f"UF ou estado: {self.state.strip()}")
        if self.activity:
            criteria.append(f"atividade: {self.activity}")
        if self.profile:
            criteria.append(f"perfil: {self.profile}")
        if self.period:
            criteria.append(f"período: {self.period}")
        if not criteria:
            return question.strip()
        return (
            f"{question.strip()}\n\n"
            f"Considere também estes filtros explícitos: {'; '.join(criteria)}."
        )
