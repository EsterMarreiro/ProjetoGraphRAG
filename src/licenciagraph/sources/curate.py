"""Resolução de códigos e verificação de integridade do manual."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from licenciagraph.sources.models import Checklist, ManualData

GENERIC_INDUSTRIAL = {
    "group": "72",
    "by_modality": {
        "LP": ["AI-01", "AI-02"],
        "LI": ["AI-03"],
        "RLI": ["AI-04"],
        "LAI": ["AI-05"],
        "LO": ["AI-06"],
        "RLO": ["AI-07"],
        "LAO": ["AI-08"],
        "LOC": ["AI-09"],
    },
}


# Títulos ausentes ou truncados no PDF; a modalidade vem da coluna ou do vizinho.
MODALITY_ERRATA = {"LM-42": "AA", "LM-13.1": "RLO"}


def checklist_key(item: Checklist, duplicated: set[str]) -> str:
    key = item.code
    if item.code in duplicated:
        key = f"{key}@{item.modality}"
    if item.variant:
        slug = re.sub(r"\W+", "-", item.variant.lower()).strip("-")
        key = f"{key}#{slug}"
    return key


def curate_manual(data: ManualData) -> dict[str, Any]:
    by_code: dict[str, list[Checklist]] = defaultdict(list)
    for item in data.checklists:
        by_code[item.code].append(item)
    for code, modality in MODALITY_ERRATA.items():
        for item in by_code.get(code, []):
            item.modality = modality
    for activity in data.activities:
        for modality, code in activity.by_modality.items():
            candidates = by_code.get(code, [])
            if len(candidates) == 1 and not candidates[0].modality:
                candidates[0].modality = modality
    duplicated = {
        code
        for code, items in by_code.items()
        if len({(i.modality, i.variant) for i in items}) > 1
        and len({i.modality for i in items}) > 1
    }
    keys = {id(item): checklist_key(item, duplicated) for item in data.checklists}
    checklists = []
    for item in data.checklists:
        checklists.append(
            {
                "key": keys[id(item)],
                "code": item.code,
                "family": item.family,
                "modality": item.modality,
                "variant": item.variant,
                "heading": item.heading,
                "pages": [item.first_page, item.last_page],
                "items": [{"number": i.number, "text": i.text} for i in item.items],
                "notes": item.notes,
                "annex": item.annex,
            }
        )
    activities = []
    for activity in data.activities:
        resolved: dict[str, list[str]] = {}
        for modality, code in activity.by_modality.items():
            matches = [
                keys[id(i)] for i in by_code.get(code, []) if i.modality == modality
            ]
            resolved[modality] = matches or [keys[id(i)] for i in by_code.get(code, [])]
        listed = [
            keys[id(i)]
            for code in activity.checklists
            if code not in activity.by_modality.values()
            for i in by_code.get(code, [])
        ]
        activities.append(
            {
                "code": activity.code,
                "name": activity.name,
                "group": [activity.group_code, activity.group_name],
                "subgroup": [activity.subgroup_code, activity.subgroup_name],
                "regime": activity.regime,
                "page": activity.page,
                "checklists_by_modality": resolved,
                "checklists": sorted(
                    {*listed, *(k for ks in resolved.values() for k in ks)}
                ),
            }
        )
    return {
        "source": data.source,
        "checklists": checklists,
        "activities": activities,
        "generic_checklists": GENERIC_INDUSTRIAL,
        "report": integrity_report(data, keys),
    }


def integrity_report(data: ManualData, keys: dict[int, str]) -> dict[str, Any]:
    codes = Counter(item.code for item in data.checklists)
    referenced = {code for a in data.activities for code in a.checklists}
    generic = {c for cs in GENERIC_INDUSTRIAL["by_modality"].values() for c in cs}
    return {
        "checklists": len(data.checklists),
        "activities": len(data.activities),
        "items": sum(len(item.items) for item in data.checklists),
        "families": dict(sorted(Counter(i.family for i in data.checklists).items())),
        "modalities": dict(
            sorted(Counter(i.modality for i in data.checklists).items())
        ),
        "without_modality": [i.code for i in data.checklists if not i.modality],
        "duplicated_codes": sorted(c for c, n in codes.items() if n > 1),
        "referenced_without_page": sorted(referenced - set(codes)),
        "pages_without_reference": sorted(set(codes) - referenced - generic),
    }


def curate_na101(
    pdf_tables: list,
    letters: dict[str, dict[str, str]],
    base_criteria: dict[str, list[dict[str, str]]],
    listed: list,
    manual: dict[str, Any],
    master: list,
) -> dict[str, Any]:
    tables = [t.to_dict() for t in pdf_tables]
    table_codes = {c for t in tables for c in t["codes"]}
    manual_codes = {a["code"] for a in manual["activities"]}
    listed_codes = {a.code for a in listed}
    return {
        "source": "na-101-5a-edicao-2026",
        "cost_letters_ufr": letters,
        "base_criteria": base_criteria,
        "porte_tables": tables,
        "master": [
            {
                "code": a.code,
                "name": a.name,
                "group_code": a.group_code,
                "group_name": a.group_name,
                "subgroup_code": a.subgroup_code,
                "subgroup_name": a.subgroup_name,
            }
            for a in master
        ],
        "listed_activities": [
            {
                "regime": a.regime,
                "code": a.code,
                "name": a.name,
                "validity": a.validity,
                "letter": a.letter,
                "page": a.page,
            }
            for a in listed
        ],
        "report": {
            "porte_tables": len(tables),
            "tables_by_rows": dict(
                sorted(Counter(len(t["rows"]) for t in tables).items())
            ),
            "listed_by_regime": dict(Counter(a.regime for a in listed)),
            "manual_activities_without_porte_table": sorted(manual_codes - table_codes)[
                :40
            ],
            "manual_without_porte_count": len(manual_codes - table_codes),
            "listed_without_porte_table": sorted(listed_codes - table_codes)[:40],
            "porte_tables_without_manual_activity_count": len(
                table_codes - manual_codes - listed_codes
            ),
        },
    }
