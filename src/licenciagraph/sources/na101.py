"""Leitor das tabelas de porte, potencial poluidor e custos da NA-101."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

ANNEX_J_FIRST_PAGE = 72
ANNEX_L_FIRST_PAGE = 383
HEADER_PREFIXES = {
    "PEQU": "PEQUENO",
    "MÉDI": "MÉDIO",
    "GRAN": "GRANDE",
    "AR": "AR",
    "ÁGUA": "ÁGUA",
    "SOLO": "SOLO",
}
PORTE_LABELS = {
    "MICRO": "MICRO",
    "PEQUENO": "PEQUENO",
    "MÉDIO": "MEDIO",
    "GRANDE": "GRANDE",
    "EXTRAORDINÁRIO": "EXTRAORDINARIO",
    "EXTRAORDINÁRI": "EXTRAORDINARIO",
    "EXTRAORDIN": "EXTRAORDINARIO",
    "EXCEPCIONAL": "EXTRAORDINARIO",
}
CODE_WORD = re.compile(r"^(?:([A-Z]{2,4}):)?(\d{2}\.\d{1,2}\.\d{1,3})$")
LETTER_WORD = re.compile(r"^[A-Z]$")
POTENTIAL_WORD = re.compile(r"^[PMG-]$")


@dataclass(slots=True)
class PorteRow:
    porte: str
    criterion: str
    letter: str
    letter_class: str
    annotation: str
    air: str
    water: str
    soil: str


@dataclass(slots=True)
class PorteTable:
    codes: list[str]
    alt_codes: dict[str, str]
    name: str
    criterion_label: str
    pages: list[int]
    rows: list[PorteRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def _words(page: pymupdf.Page) -> list[_Word]:
    return [
        _Word(x0, y0, x1, y1, text)
        for x0, y0, x1, y1, text, *_ in page.get_text("words")
    ]


def _horizontal_lines(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    lines = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.height < 2 and rect.width > 15:
            lines.append((rect.y0, rect.x0, rect.x1))
    return lines


def parse_porte_tables(path: Path) -> list[PorteTable]:
    document = pymupdf.open(path)
    tables: list[PorteTable] = []
    current: PorteTable | None = None
    columns: dict[str, list[float]] = {}
    for index in range(ANNEX_J_FIRST_PAGE - 1, ANNEX_L_FIRST_PAGE - 1):
        page = document[index]
        number = index + 1
        words = [w for w in _words(page) if 40 < w.y0 < 790]
        lines = _horizontal_lines(page)
        anchors = sorted(
            w.y0 - 6
            for w in words
            if w.text.upper() in {"CÓDIGO", "CODIGO"}
            and w.x0 < 130
            and any(
                v.text == "DA" and -3 < v.y0 - w.y0 < 20 and abs(v.x0 - w.x0) < 45
                for v in words
            )
        )
        bounds = [40.0] if not anchors or anchors[0] > 60 else []
        bounds += anchors
        bounds.append(800.0)
        for k in range(len(bounds) - 1):
            region = [w for w in words if bounds[k] <= w.y0 < bounds[k + 1]]
            is_start = bounds[k] in anchors
            if is_start:
                current = _new_table(region, number)
                tables.append(current)
                columns = _header_columns(region)
            elif current is not None and not columns and _header_columns(region):
                current.pages.append(number)
                columns = _header_columns(region)
                head = _new_table(region, number)
                current.codes = current.codes or head.codes
                current.alt_codes = current.alt_codes or head.alt_codes
            elif _continues_table(region, columns):
                if current is not None and number not in current.pages:
                    current.pages.append(number)
                columns = {**columns, "header_y": [0.0]}
            else:
                continue
            if current is None:
                continue
            if columns:
                current.rows.extend(_read_rows(region, lines, columns))
            current.notes.extend(_read_notes(region))
    return tables


def _continues_table(region: list[_Word], columns: dict[str, list[float]]) -> bool:
    if not columns:
        return False
    limit = columns["label_limit"][0]
    head = [w for w in region if w.y0 < 330]
    return any(w.text in PORTE_LABELS and w.cx < limit for w in head)


def _new_table(region: list[_Word], page: int) -> PorteTable:
    codes: list[str] = []
    alt: dict[str, str] = {}
    for word in region:
        match = CODE_WORD.match(word.text)
        if match and word.x0 < 175:
            code = match.group(2)
            if match.group(1):
                alt[match.group(1)] = code
            elif code.startswith("49."):
                alt.setdefault("LAU", code)
            elif code not in codes:
                codes.append(code)
    header_y = _header_y(region)
    name_words = [
        w
        for w in region
        if w.x0 >= 190
        and w.x1 <= 470
        and w.y0 < (header_y if header_y else 10_000) - 20
        and w.text
        not in {"POTENCIAL", "POLUIDOR", "POLUIDOR/DEGRADADOR", "DA", "ATIVIDADE"}
    ]
    name_words.sort(key=lambda w: (round(w.y0 / 4), w.x0))
    criterion = [
        w.text
        for w in sorted(region, key=lambda w: (w.y0, w.x0))
        if w.x0 < 165
        and header_y
        and w.y0 > header_y
        and w.text not in {"CRITÉRIO", "DE", "PORTE:"}
    ]
    return PorteTable(
        codes=codes,
        alt_codes=alt,
        name=" ".join(w.text for w in name_words).strip(),
        criterion_label=" ".join(criterion).strip(),
        pages=[page],
    )


def _header_y(region: list[_Word]) -> float | None:
    for word in region:
        if word.text == "AR" and word.x0 > 400:
            return word.y0
    return None


def _header_name(text: str) -> str:
    if text in {"AR", "ÁGUA", "SOLO"}:
        return text
    if len(text) >= 4:
        for prefix, name in HEADER_PREFIXES.items():
            if len(prefix) >= 4 and text.startswith(prefix) and len(text) <= 8:
                return name
    return ""


def _header_columns(region: list[_Word]) -> dict[str, list[float]]:
    y = _header_y(region)
    if y is None:
        return {}
    names = ("PEQUENO", "MÉDIO", "GRANDE", "AR", "ÁGUA", "SOLO")
    found: dict[str, _Word] = {}
    for word in region:
        name = _header_name(word.text)
        if name and word.x0 > 230 and abs(word.y0 - y) < 22:
            best = found.get(name)
            if best is None or abs(word.y0 - y) < abs(best.y0 - y):
                found[name] = word
    if len(found) < 6:
        return {}
    return {
        "letters": [found[n].cx for n in names[:3]],
        "potential": [found[n].cx for n in names[3:]],
        "label_limit": [found["PEQUENO"].cx - 60],
        "header_y": [max(w.y0 for w in found.values())],
    }


def _read_rows(
    region: list[_Word],
    lines: list[tuple[float, float, float]],
    columns: dict[str, list[float]],
) -> list[PorteRow]:
    header_y = columns["header_y"][0]
    limit = columns["label_limit"][0]
    labels = sorted(
        (
            w
            for w in region
            if w.text in PORTE_LABELS and w.y0 > header_y + 5 and w.cx < limit
        ),
        key=lambda w: w.y0,
    )
    if not labels:
        return []
    label_x = labels[0].x0
    ys = sorted(
        {
            round(y)
            for y, x0, x1 in lines
            if y > header_y and x0 <= label_x + 5 and x1 >= label_x + 30
        }
    )
    rows = []
    for label in labels:
        top = max([y for y in ys if y <= label.y0 + 1] or [label.y0 - 6])
        below = [y for y in ys if y > label.y0 + 4]
        bottom = min(below) if below else label.y1 + 6
        band = [w for w in region if top - 1 <= w.cy <= bottom + 1 and w is not label]
        rows.append(_read_row(label, band, columns, limit))
    return rows


def _read_row(
    label: _Word, band: list[_Word], columns: dict[str, list[float]], limit: float
) -> PorteRow:
    letters = columns["letters"]
    potentials = columns["potential"]
    criterion: list[str] = []
    letter = ""
    letter_class = ""
    annotation: list[str] = []
    potential = {"AR": "", "ÁGUA": "", "SOLO": ""}
    centers = [*letters, *potentials]
    for word in sorted(band, key=lambda w: (w.y0, w.x0)):
        if word.cx < limit + 40 and word.x0 > label.x1 + 2:
            criterion.append(word.text)
            continue
        distances = [abs(word.cx - c) for c in centers]
        nearest = distances.index(min(distances))
        if min(distances) > 40:
            continue
        if nearest < 3:
            if LETTER_WORD.match(word.text):
                letter = word.text
                letter_class = ("PEQUENO", "MEDIO", "GRANDE")[nearest]
            elif word.text.startswith("("):
                annotation.append(word.text)
            elif word.text.endswith(")") or word.text == "/RIMA)":
                annotation.append(word.text)
        elif POTENTIAL_WORD.match(word.text):
            potential[("AR", "ÁGUA", "SOLO")[nearest - 3]] = word.text
    return PorteRow(
        porte=PORTE_LABELS[label.text],
        criterion=" ".join(criterion),
        letter=letter,
        letter_class=letter_class,
        annotation=" ".join(annotation),
        air=potential["AR"],
        water=potential["ÁGUA"],
        soil=potential["SOLO"],
    )


def _read_notes(region: list[_Word]) -> list[str]:
    starts = [w for w in region if w.text.startswith("OBS")]
    notes = []
    for start in starts:
        text = [
            w.text
            for w in sorted(region, key=lambda w: (round(w.y0 / 4), w.x0))
            if w.y0 >= start.y0 - 1 and w.x0 >= start.x0 - 1
        ]
        notes.append(" ".join(text))
    return notes


COST_COLUMNS = (
    "LP",
    "LI",
    "LP+LI",
    "LO",
    "LP+LI+LO",
    "LOC",
    "LOP",
    "LAU",
    "LAC",
    "LAI_LAO",
    "AA_LTE",
    "LVPE",
    "REGISTRO",
)
COST_PAGE = 19
COST_CENTERS = (83, 136, 192, 246, 304, 372, 439, 498, 563, 629, 690, 748, 801)


def parse_cost_letters(path: Path) -> dict[str, dict[str, str]]:
    """Matriz letra x modalidade do Anexo B, em UFR-PB."""
    page = pymupdf.open(path)[COST_PAGE - 1]
    words = [w for w in _words(page) if w.y0 > 150 and w.y0 < 395]
    labels = sorted(
        (w for w in words if w.x0 < 60 and re.fullmatch(r"[A-Z]", w.text)),
        key=lambda w: w.y0,
    )
    table: dict[str, dict[str, str]] = {label.text: {} for label in labels}
    for word in words:
        if word.x0 < 60 or word.text in {"LP", "+", "LI"}:
            continue
        label = min(labels, key=lambda w: abs(w.cy - word.cy))
        distances = [abs(word.cx - c) for c in COST_CENTERS]
        if abs(label.cy - word.cy) > 12 or min(distances) > 30:
            continue
        table[label.text][COST_COLUMNS[distances.index(min(distances))]] = word.text
    if table.get("A", {}).get("LP") != "4" or table.get("F", {}).get("LI") != "19":
        raise ValueError("Anexo B fora do layout esperado")
    return {letter: cells for letter, cells in table.items() if cells}


LIST_ANNEXES = {
    "dispensa": (20, 25),
    "lac": (26, 30),
    "lau": (31, 34),
    "aa": (35, 37),
    "lte": (38, 38),
    "lvpe": (39, 39),
}
LIST_GROUP = re.compile(r"^\d{2}$")
LIST_SUBGROUP = re.compile(r"^\d{2}\.\d{2}$")
LIST_ACTIVITY = re.compile(r"^\d{2}\.\d{1,3}(?:\.\d{1,3})?$")
LIST_TAIL = re.compile(
    r"^(?P<name>.*?)\s*(?P<validity>\d+\s+ANOS?|½\s+ANO|De acordo com o cronograma)?"
    r"\s*(?P<letter>[A-Z])?$"
)


@dataclass(slots=True)
class ListedActivity:
    regime: str
    code: str
    name: str
    validity: str
    letter: str
    page: int


def parse_activity_lists(path: Path) -> list[ListedActivity]:
    document = pymupdf.open(path)
    results: list[ListedActivity] = []
    for regime, (first, last) in LIST_ANNEXES.items():
        rows: list[tuple[int, str]] = []
        for number in range(first, last + 1):
            for raw in document[number - 1].get_text().split("\n"):
                line = raw.strip()
                if line and not re.fullmatch(r"\d+", line) or LIST_GROUP.match(line):
                    rows.append((number, line))
        results.extend(_activities_from_rows(regime, rows))
    return results


def _activities_from_rows(
    regime: str, rows: list[tuple[int, str]]
) -> list[ListedActivity]:
    found: list[ListedActivity] = []
    code = ""
    page = 0
    chunk: list[str] = []

    def flush() -> None:
        if not code or not chunk:
            return
        text = " ".join(chunk).replace("  ", " ").strip()
        match = LIST_TAIL.match(text)
        if not match:
            return
        name = match.group("name").strip(" -*")
        if not name or "não previstas" in name:
            return
        found.append(
            ListedActivity(
                regime=regime,
                code=code,
                name=name,
                validity=(match.group("validity") or "").strip(),
                letter=match.group("letter") or "",
                page=page,
            )
        )

    for number, line in rows:
        if line in {
            "CÓDIGO",
            "GRUPO/ATIVIDADE",
            "PRAZOS",
            "LETRA DE",
            "REMUNERAÇÃO",
            "-",
        }:
            continue
        if LIST_ACTIVITY.match(line) and not LIST_SUBGROUP.match(line):
            flush()
            code, page, chunk = line, number, []
        elif LIST_SUBGROUP.match(line) or LIST_GROUP.match(line):
            flush()
            code, chunk = "", []
        elif code:
            chunk.append(line)
    flush()
    return found


BASE_CRITERIA_PAGE = 18
PORTES = ("MICRO", "PEQUENO", "MEDIO", "GRANDE", "EXTRAORDINARIO")


def parse_base_criteria(path: Path) -> dict[str, list[dict[str, str]]]:
    """Critérios-base de porte (área, faturamento em UFR-PB, funcionários)."""
    page = pymupdf.open(path)[BASE_CRITERIA_PAGE - 1]
    lines = [line.strip() for line in page.get_text().split("\n") if line.strip()]
    tables: dict[str, list[dict[str, str]]] = {}
    current = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("CRITÉRIOS BASES"):
            current = line.replace("CRITÉRIOS BASES", "").strip()
            tables[current] = []
        elif current and _plain(line) in PORTES:
            cells = lines[index + 1 : index + 4]
            tables[current].append(
                {
                    "porte": _plain(line),
                    "area_m2": cells[0],
                    "faturamento_ufr": cells[1],
                    "funcionarios": cells[2],
                }
            )
            index += 3
        index += 1
    return tables


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).upper()


def prose_pages(path: Path, first: int, last: int) -> list[dict[str, Any]]:
    """Texto corrido por página, para a extração por LLM."""
    document = pymupdf.open(path)
    pages = []
    for number in range(first, last + 1):
        text = "\n".join(
            line.strip()
            for line in document[number - 1].get_text().split("\n")
            if line.strip()
        )
        pages.append({"page": number, "text": text})
    return pages
