"""Leitor do Anexo A do Manual de Licenciamento Ambiental da SUDEMA."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pymupdf

from licenciagraph.sources.models import (
    Activity,
    Checklist,
    ChecklistItem,
    ManualData,
)

PAGE_HEADER = (
    "Governo do Estado",
    "Secretaria de Estado",
    "Superintendência de Adm",
)
ANNEX_FIRST_PAGE = 32
SECTION_REGIME = {
    1: "dispensa",
    2: "lac",
    3: "lau",
    4: "aa",
    5: "florestal",
}
CHECKLIST_CODE = r"[A-Z]{1,5}\s*[–-]?\s*\d{1,2}(?:\.\d)?"
CHECKLIST_START = re.compile(rf"^({CHECKLIST_CODE})\s*(?:[–-]\s*(.*)|(\(.*\)))?$")
CHECKLIST_ONLY = re.compile(rf"^(?:{CHECKLIST_CODE}\s*)+$")
SECTION_START = re.compile(r"^(\d{1,2})\.\s*([A-Za-zÀ-ú].*)$")
GROUP_LINE = re.compile(r"^\d{2}$")
SUBGROUP_LINE = re.compile(r"^\d{2}\.\d{2}$")
ACTIVITY_LINE = re.compile(r"^(?:\d{2}\.\d{2}\.\d{1,3}|\d{2}\.\d{3})$")
NOISE_LINES = {
    "CÓDIGO",
    "CÓDIG",
    "O",
    "GRUPO/ATIVIDADE",
    "CHECK",
    "LISTS",
    "CHECK - LISTS",
    "CHECK LISTS",
    "CHECK-LISTS",
    "CHECK -",
    "-",
    "LP",
    "LI",
    "LO",
    "RLI",
    "RLO",
    "LAI",
    "LAO",
    "LOC",
    "AA",
    "LTE",
    "ENC",
    "ENC.",
    "C.",
    "Registro",
    "REGISTRO",
}
BOILERPLATE = (
    "As atividades pertencentes a este grupo",
    "Em casos de abertura de processos",
)
MODALITY_PATTERNS = (
    ("LAC", r"ADES[ÃA]O E COMPROMISSO"),
    ("LAU", r"AMBIENTAL [ÚU]NICA"),
    ("LTE", r"TRANSPORTE ESTADUAL|TRASPORTE ESTADUAL"),
    ("LVPE", r"VEICULAR DE PROPAGANDA"),
    ("AA", r"AUTORIZA[ÇC][ÃA]O AMBIENTAL"),
    ("RLOP", r"RENOVA[ÇC][ÃA]O.*OPERA[ÇC][ÃA]O PARA PESQUISA"),
    ("LOP", r"OPERA[ÇC][ÃA]O PARA PESQUISA"),
    ("RLI", r"RENOVA[ÇC][ÃA]O.*INSTALA[ÇC][ÃA]O"),
    ("RLO", r"RENOVA[ÇC][ÃA]O.*OPERA[ÇC][ÃA]O"),
    ("LAI", r"ALTERA[ÇC][ÃA]O.*INSTALA[ÇC][ÃA]O"),
    ("LAO", r"ALTERA[ÇC][ÃA]O.*OPERA[ÇC][ÃA]O"),
    ("LOC", r"OPERA[ÇC][ÃA]O CORRETIVA"),
    ("LIO", r"INSTALA[ÇC][ÃA]O E OPERA[ÇC][ÃA]O"),
    ("LP", r"LICEN[ÇC]A PR[ÉE]VIA"),
    ("LI", r"LICEN[ÇC]A DE INSTALA[ÇC][ÃA]O"),
    ("LO", r"LICEN[ÇC]A DE OPERA[ÇC][ÃA]O"),
    ("ENC", r"ENCERRAMENTO"),
    ("COMPENSACAO", r"COMPENSA[ÇC][ÃA]O"),
    ("REGISTRO", r"^REGISTRO$"),
)


def normalize_code(raw: str) -> str:
    match = re.match(r"([A-Z]{1,5})\s*[–-]?\s*(\d{1,2}(?:\.\d)?)", raw.strip())
    if not match:
        raise ValueError(f"codigo de checklist invalido: {raw!r}")
    return f"{match.group(1)}-{match.group(2).zfill(2)}"


def classify_modality(heading: list[str]) -> str:
    text = unicodedata.normalize("NFC", " ".join(heading).upper())
    for modality, pattern in MODALITY_PATTERNS:
        if re.search(pattern, text):
            return modality
    return ""


def _page_lines(page: pymupdf.Page, number: int) -> list[str]:
    lines: list[str] = []
    page_number_dropped = False
    for raw in page.get_text().split("\n"):
        line = raw.strip()
        if not line or line.startswith(PAGE_HEADER):
            continue
        if not page_number_dropped and re.fullmatch(r"\d+", line):
            if int(line) == number:
                page_number_dropped = True
                continue
        lines.append(line)
    return lines


FIRST_ITEM = re.compile(r"^1\s+(requerimento.*)$", re.IGNORECASE)


def _first_item_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines[:60]):
        if FIRST_ITEM.match(line):
            return index
        if line == "1" and index + 1 < len(lines):
            if lines[index + 1].lower().startswith("requerimento"):
                return index
    return None


def _is_checklist_page(lines: list[str]) -> bool:
    first = _first_item_index(lines)
    if first is None or not lines:
        return False
    if CHECKLIST_START.match(lines[0]):
        return True
    return first <= 8 and classify_modality(lines[:first]) != ""


def _section_number(lines: list[str]) -> int | None:
    match = SECTION_START.match(lines[0]) if lines else None
    if not match or not any("CÓDIG" in line for line in lines[:8]):
        return None
    return int(match.group(1))


def parse_manual(path: Path) -> ManualData:
    document = pymupdf.open(path)
    activities: list[Activity] = []
    checklists: list[Checklist] = []
    regime = ""
    stream: list[tuple[int, str]] = []
    table_streams: list[tuple[str, list[tuple[int, str]]]] = []
    table: list[tuple[int, str]] = []
    for index in range(ANNEX_FIRST_PAGE - 1, len(document)):
        number = index + 1
        lines = _page_lines(document[index], number)
        section = _section_number(lines)
        if section is not None:
            checklists.extend(_parse_stream(stream))
            stream = []
            if table:
                table_streams.append((regime, table))
            table = []
            regime = SECTION_REGIME.get(section, "licenca")
            lines = lines[1:]
        if not stream and not _is_checklist_page(lines):
            table.extend((number, line) for line in lines)
            continue
        stream.extend((number, line) for line in lines)
    checklists.extend(_parse_stream(stream))
    if table:
        table_streams.append((regime, table))
    for section_regime, rows in table_streams:
        activities.extend(_parse_table(rows, section_regime))
    return ManualData(activities=_merge_activities(activities), checklists=checklists)


NOTE_MARKERS = ("Poderão ser solicitados", "A SUDEMA em qualquer momento")
CODE_LOOKBACK = 40


def _parse_stream(stream: list[tuple[int, str]]) -> list[Checklist]:
    lines = [line for _, line in stream]
    pages = [page for page, _ in stream]
    starts = [i for i in range(len(lines)) if _is_first_item(lines, i)]
    first_numbers = dict.fromkeys(starts, 1)
    for index, start in _orphan_headings(lines, starts):
        starts.append(start)
        first_numbers[start] = index
    starts.sort()
    headings: list[int] = []
    previous = -1
    for start in starts:
        found = start
        for back in range(start - 1, max(previous, start - CODE_LOOKBACK) - 1, -1):
            if back > previous and CHECKLIST_START.match(lines[back]):
                found = back
                break
        if found == start and start > 0 and lines[start - 1].endswith(":"):
            found = start - 1
        headings.append(found)
        previous = start
    results: list[Checklist] = []
    for k, start in enumerate(starts):
        end = headings[k + 1] if k + 1 < len(starts) else len(lines)
        head = headings[k]
        match = CHECKLIST_START.match(lines[head]) if head != start else None
        code = normalize_code(match.group(1)) if match else ""
        variant = ""
        if not match and head != start and results:
            variant = lines[head].rstrip(":").strip()
            code = results[-1].code
        title = [match.group(2).strip()] if match and match.group(2) else []
        title += lines[head + 1 : start]
        carried = _split_carried_item(title)
        if carried and results:
            _append_item(results[-1], carried)
        checklist = Checklist(
            code=code,
            family=code.split("-")[0] if code else "",
            modality=classify_modality(title)
            or (results[-1].modality if variant else ""),
            heading=title,
            variant=variant,
            first_page=pages[head],
            last_page=pages[max(end - 1, head)],
        )
        _read_items(checklist, lines[start:end], pages[start:end], first_numbers[start])
        results.append(checklist)
    return results


def _orphan_headings(lines: list[str], starts: list[int]) -> list[tuple[int, int]]:
    """Checklists cujo item 1 foi omitido no PDF: devolve (numero inicial, linha)."""
    found = []
    for index, line in enumerate(lines):
        if (
            not CHECKLIST_START.match(line)
            or classify_modality(lines[index + 1 : index + 3]) == ""
        ):
            continue
        following = [s for s in starts if s > index]
        window = lines[index + 1 : (following[0] if following else len(lines))]
        if any(
            _CODE_AND_MODALITY(lines, j)
            for j in range(index + 1, index + 1 + len(window))
        ):
            window = window[
                : next(
                    j
                    for j in range(len(window))
                    if _CODE_AND_MODALITY(lines, index + 1 + j)
                )
            ]
        if (
            following
            and following[0] - index <= 30
            and not _has_other_code(lines, index, following[0])
        ):
            continue
        for offset, candidate in enumerate(window[:6]):
            if re.fullmatch(r"\d{1,2}", candidate):
                found.append((int(candidate), index + 1 + offset))
                break
    return found


def _CODE_AND_MODALITY(lines: list[str], index: int) -> bool:
    return bool(
        CHECKLIST_START.match(lines[index])
        and classify_modality(lines[index + 1 : index + 3])
    )


def _has_other_code(lines: list[str], index: int, start: int) -> bool:
    return any(CHECKLIST_START.match(line) for line in lines[index + 1 : start])


CARRIED_ITEM = re.compile(r"^(\d{1,2})\s+\S")


def _split_carried_item(title: list[str]) -> list[str]:
    """Tira do título as linhas de um item deslocado pelo PDF para após o código."""
    for index, line in enumerate(title):
        if CARRIED_ITEM.match(line):
            carried = title[index:]
            del title[index:]
            return carried
    return []


def _append_item(checklist: Checklist, lines: list[str]) -> None:
    match = CARRIED_ITEM.match(lines[0])
    assert match is not None
    number = int(match.group(1))
    text = "\n".join([lines[0][match.end(1) :].strip(), *lines[1:]])
    if number == len(checklist.items) + 1:
        checklist.items.append(ChecklistItem(number=number, text=text))
    else:
        checklist.notes.append(" ".join(lines))


def _is_first_item(lines: list[str], index: int) -> bool:
    if FIRST_ITEM.match(lines[index]):
        return True
    return (
        lines[index] == "1"
        and index + 1 < len(lines)
        and lines[index + 1].lower().startswith("requerimento")
    )


def _read_items(
    checklist: Checklist, lines: list[str], pages: list[int], first_number: int = 1
) -> None:
    closed_page = 0
    annex: dict[int, list[str]] = {}
    for line, page in zip(lines, pages, strict=True):
        if closed_page:
            if page == closed_page:
                checklist.notes.append(line)
            else:
                annex.setdefault(page, []).append(line)
            continue
        if line.startswith(NOTE_MARKERS):
            closed_page = page
            checklist.notes.append(line)
            continue
        expected = first_number + len(checklist.items)
        same_line = re.match(rf"^{expected}\s+(\S.*)$", line)
        if line == str(expected):
            checklist.items.append(ChecklistItem(number=expected, text=""))
        elif same_line and (not checklist.items or checklist.items[-1].text):
            checklist.items.append(ChecklistItem(expected, same_line.group(1)))
        elif checklist.items:
            item = checklist.items[-1]
            item.text = f"{item.text}\n{line}".strip()
    checklist.notes = [" ".join(checklist.notes)] if checklist.notes else []
    checklist.annex = ["\n".join(block) for block in annex.values()]


def _dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


COLUMN_TOKENS = (
    "RLOP", "LVPE", "RLI", "RLO", "LAI", "LAO", "LOC", "LOP", "LTE", "ENC",
    "LP", "LI", "LO", "AA",
)  # fmt: skip
HEADER_FRAGMENT = re.compile(r"^[A-Z.]{1,5}$")


def parse_columns(fragments: list[str]) -> list[str]:
    for start in range(len(fragments)):
        text = "".join(fragments[start:]).replace(".", "")
        columns: list[str] = []
        while text:
            token = next((t for t in COLUMN_TOKENS if text.startswith(t)), None)
            if token is None:
                break
            columns.append(token)
            text = text[len(token) :]
        if not text and columns:
            return columns
    return []


def _parse_table(rows: list[tuple[int, str]], regime: str) -> list[Activity]:
    results: list[Activity] = []
    group = ("", "")
    subgroup = ("", "")
    columns: list[str] = []
    fragments: list[str] = []
    pending: dict | None = None
    mode = ""

    def close() -> None:
        nonlocal pending
        name = " ".join(pending["name"]) if pending else ""
        if pending and name and "não previstas" not in name:
            cells = pending["cells"]
            by_modality = {}
            if columns and len(cells) == len(columns):
                by_modality = {c: v for c, v in zip(columns, cells, strict=True) if v}
            results.append(
                Activity(
                    code=pending["code"],
                    name=name.strip(),
                    group_code=group[0],
                    group_name=group[1],
                    subgroup_code=subgroup[0],
                    subgroup_name=subgroup[1],
                    regime=regime,
                    checklists=_dedupe([c for c in cells if c]),
                    page=pending["page"],
                    by_modality=by_modality,
                )
            )
        pending = None

    for page, line in rows:
        if line.startswith(BOILERPLATE):
            continue
        if mode == "group" and HEADER_FRAGMENT.match(line):
            fragments.append(line)
            columns = parse_columns(fragments)
            continue
        if ACTIVITY_LINE.match(line):
            close()
            pending = {"code": line, "name": [], "cells": [], "page": page}
            mode = "activity"
        elif SUBGROUP_LINE.match(line):
            close()
            subgroup = (line, "")
            mode = "subgroup"
        elif GROUP_LINE.match(line):
            close()
            group = (line, "")
            subgroup = ("", "")
            columns = []
            fragments = []
            mode = "group"
        elif CHECKLIST_ONLY.match(line) and pending is not None:
            for match in re.finditer(CHECKLIST_CODE, line):
                pending["cells"].append(normalize_code(match.group(0)))
        elif line == "-" and mode == "activity" and pending is not None:
            pending["cells"].append(None)
        elif line in NOISE_LINES:
            continue
        elif mode == "activity" and pending is not None:
            pending["name"].append(line)
        elif mode == "subgroup":
            subgroup = (subgroup[0], f"{subgroup[1]} {line}".strip())
        elif mode == "group":
            group = (group[0], f"{group[1]} {line}".strip())
    close()
    return results


def _merge_activities(activities: list[Activity]) -> list[Activity]:
    merged: dict[str, Activity] = {}
    for activity in activities:
        key = activity.code
        known = merged.get(key)
        if known is None:
            merged[key] = activity
            continue
        known.checklists = _dedupe([*known.checklists, *activity.checklists])
        known.by_modality = {**known.by_modality, **activity.by_modality}
        if activity.regime not in known.regime.split("+"):
            known.regime = f"{known.regime}+{activity.regime}"
    return list(merged.values())
