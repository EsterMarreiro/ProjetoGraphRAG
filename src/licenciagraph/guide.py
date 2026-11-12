"""Assistente de enquadramento: consultas diretas ao grafo, sem LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PORTE_ORDER = ("MICRO", "PEQUENO", "MEDIO", "GRANDE", "EXTRAORDINARIO")
PORTE_LABELS = {
    "MICRO": "Micro",
    "PEQUENO": "Pequeno",
    "MEDIO": "Médio",
    "GRANDE": "Grande",
    "EXTRAORDINARIO": "Extraordinário",
}
NUMBER = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?"
BOUND = re.compile(rf"(≤|≥|<=|>=|<|>)\s*({NUMBER})")


@dataclass(frozen=True, slots=True)
class Interval:
    low: float | None = None
    low_inclusive: bool = False
    high: float | None = None
    high_inclusive: bool = False

    def contains(self, value: float) -> bool:
        if self.low is not None:
            if value < self.low or (value == self.low and not self.low_inclusive):
                return False
        if self.high is not None:
            if value > self.high or (value == self.high and not self.high_inclusive):
                return False
        return True


def parse_number(text: str) -> float:
    return float(text.replace(".", "").replace(",", "."))


def parse_interval(criterion: str) -> Interval | None:
    """Interpreta critérios como '>3 ≤20', '≥1<2,5' ou '> 100'."""
    low = high = None
    low_inclusive = high_inclusive = False
    found = False
    for operator, number in BOUND.findall(criterion or ""):
        found = True
        value = parse_number(number)
        if operator in (">", "≥", ">="):
            low, low_inclusive = value, operator != ">"
        else:
            high, high_inclusive = value, operator in ("≤", "<=")
    if not found:
        return None
    return Interval(low, low_inclusive, high, high_inclusive)


@dataclass(frozen=True, slots=True)
class ActivityOption:
    id: str
    code: str
    name: str
    group: str


@dataclass(frozen=True, slots=True)
class Band:
    porte: str
    criterion: str
    letter: str
    potential_class: str
    air: str
    water: str
    soil: str
    note: str

    @property
    def label(self) -> str:
        return PORTE_LABELS.get(self.porte, self.porte)

    @property
    def interval(self) -> Interval | None:
        return parse_interval(self.criterion)


@dataclass(frozen=True, slots=True)
class Regime:
    modality: str
    name: str
    validity: str
    letter: str


@dataclass(frozen=True, slots=True)
class ChecklistRef:
    id: str
    code: str
    modality: str
    title: str
    pages: str


@dataclass(slots=True)
class ActivityProfile:
    id: str
    code: str
    name: str
    group: str
    subgroup: str
    porte_unit: str
    regimes: list[Regime] = field(default_factory=list)
    checklists: list[ChecklistRef] = field(default_factory=list)
    bands: list[Band] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Requirement:
    number: int
    text: str
    document: str
    agency: str
    condition: str


def band_for_value(bands: list[Band], value: float) -> Band | None:
    for band in bands:
        interval = band.interval
        if interval is not None and interval.contains(value):
            return band
    return None


class LicensingGuide:
    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def _rows(self, query: str, **params: Any) -> list[list[Any]]:
        return self.graph.ro_query(query, params=params).result_set

    def search_activities(self, text: str, limit: int = 15) -> list[ActivityOption]:
        text = text.strip()
        if len(text) < 2:
            return []
        rows = self._rows(
            "MATCH (a:Atividade) WHERE toLower(a.nome) CONTAINS toLower($text) "
            "OR a.codigo STARTS WITH $text "
            "RETURN a.id, a.codigo, a.nome, a.grupo ORDER BY a.codigo LIMIT $limit",
            text=text,
            limit=limit,
        )
        return [ActivityOption(*(str(v or "") for v in row)) for row in rows]

    def profile(self, activity_id: str) -> ActivityProfile | None:
        head = self._rows(
            "MATCH (a:Atividade {id: $id}) "
            "RETURN a.nome, a.codigo, a.grupo, a.subgrupo, a.unidade_porte",
            id=activity_id,
        )
        if not head:
            return None
        name, code, group, subgroup, unit = (str(v or "") for v in head[0])
        profile = ActivityProfile(activity_id, code, name, group, subgroup, unit)
        for row in self._rows(
            "MATCH (a:Atividade {id: $id})-[r:ENQUADRADA_EM]->(m:Modalidade) "
            "RETURN m.sigla, m.nome, r.validade, r.letra ORDER BY m.sigla",
            id=activity_id,
        ):
            profile.regimes.append(Regime(*(str(v or "") for v in row)))
        for row in self._rows(
            "MATCH (a:Atividade {id: $id})-[:EXIGE_CHECKLIST]->(c:Checklist) "
            "RETURN c.id, c.codigo, c.modalidade, c.nome, c.paginas "
            "ORDER BY c.modalidade",
            id=activity_id,
        ):
            profile.checklists.append(ChecklistRef(*(str(v or "") for v in row)))
        for row in self._rows(
            "MATCH (a:Atividade {id: $id})-[:TEM_FAIXA]->(f:Faixa) "
            "RETURN f.porte, f.criterio, f.letra, f.classe, f.ar, f.agua, f.solo, "
            "f.anotacao",
            id=activity_id,
        ):
            profile.bands.append(Band(*(str(v or "") for v in row)))
        profile.bands.sort(key=lambda b: PORTE_ORDER.index(b.porte))
        return profile

    def costs(self, letter: str, modalities: list[str]) -> dict[str, str]:
        if not letter:
            return {}
        rows = self._rows(
            "MATCH (l:LetraCusto {id: $id})-[r:CUSTA]->(m:Modalidade) "
            "WHERE m.sigla IN $modalities RETURN m.sigla, r.valor_ufr",
            id=f"letracusto:{letter.lower()}",
            modalities=modalities,
        )
        return {str(sigla): str(value) for sigla, value in rows}

    def requirements(self, checklist_id: str) -> list[Requirement]:
        rows = self._rows(
            "MATCH (t:Trecho)-[:MENCIONA]->(c:Checklist {id: $id}) "
            "WHERE t.tipo = 'checklist' RETURN t.texto",
            id=checklist_id,
        )
        originals: dict[int, str] = {}
        for row in rows[:1]:
            for line in str(row[0]).split("\n"):
                match = re.match(r"^(\d+)\.\s+(.*)$", line)
                if match:
                    originals[int(match.group(1))] = match.group(2)
        details = {
            int(order): (str(name or ""), str(agency or ""), str(condition or ""))
            for order, name, condition, agency in self._rows(
                "MATCH (c:Checklist {id: $id})-[r:EXIGE_DOCUMENTO]->"
                "(d:DocumentoExigido) "
                "OPTIONAL MATCH (d)-[:EMITIDO_POR]->(o:Orgao) "
                "RETURN r.ordem, d.nome, d.condicao, o.nome",
                id=checklist_id,
            )
            if order is not None
        }
        return [
            Requirement(
                number=number,
                text=text,
                document=details.get(number, ("", "", ""))[0],
                agency=details.get(number, ("", "", ""))[1],
                condition=details.get(number, ("", "", ""))[2],
            )
            for number, text in sorted(originals.items())
        ]


def checklist_markdown(
    profile: ActivityProfile, checklist: ChecklistRef, items: list[Requirement]
) -> str:
    lines = [
        f"# {checklist.title}",
        "",
        f"Atividade: {profile.code} {profile.name}",
        f"Fonte: Manual de Licenciamento Ambiental da SUDEMA, 5ª edição, "
        f"p. {checklist.pages}.",
        "",
    ]
    lines += [f"{item.number}. {item.text}" for item in items]
    lines += [
        "",
        "Informação orientativa; confirme com a SUDEMA antes de protocolar.",
    ]
    return "\n".join(lines)
