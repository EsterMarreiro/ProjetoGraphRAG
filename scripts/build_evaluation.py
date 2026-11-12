"""Gera os conjuntos de avaliação a partir dos dados curados.

O gabarito de cada pergunta é calculado nas tabelas curadas, nunca pelo sistema
avaliado. As perguntas seguem modelos; perguntas escritas por pessoas de fora
do projeto devem complementar este conjunto.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from licenciagraph.guide import PORTE_LABELS, parse_interval
from licenciagraph.ingestion.curated import (
    MODALITIES,
    build_structured_documents,
    load_curated,
    merge_activities,
)

ROOT = Path(__file__).resolve().parents[1]
GENERIC_ITEMS = (
    "requerimento padrão",
    "cadastro devidamente",
    "documentos individuais",
    "guia de recolhimento",
)
MODALITY_QUESTION = {
    "LP": "Licença Prévia",
    "LI": "Licença de Instalação",
    "LO": "Licença de Operação",
    "RLO": "renovação da Licença de Operação",
    "LOC": "Licença de Operação Corretiva",
}


class Builder:
    def __init__(self, curated: Path) -> None:
        self.data = load_curated(curated)
        self.activities = merge_activities(self.data)
        self.checklists = {c["key"]: c for c in self.data["manual"]["checklists"]}
        self.documents = self.data["documentos"]
        self.chunk_ids: dict[str, str] = {}
        self.chunk_text: dict[str, str] = {}
        for document in build_structured_documents(curated):
            for planned in document.chunks:
                first = planned.chunk.text.split("\n", 1)[0]
                self.chunk_ids[first] = planned.chunk.id
                self.chunk_text[planned.chunk.id] = planned.chunk.text
        self.letters = self.data["na101"]["cost_letters_ufr"]

    def activity_chunk(self, code: str) -> str:
        activity = self.activities[code]
        return self._lookup(f"Atividade {code}: {activity['name']}.")

    def checklist_chunk(self, key: str) -> str:
        item = self.checklists[key]
        prefix = f"Checklist {item['code']}: {MODALITIES.get(item['modality'])}"
        return next(
            chunk
            for first, chunk in self.chunk_ids.items()
            if first.startswith(prefix)
            and (not item["variant"] or item["variant"] in first)
        )

    def _lookup(self, first_line: str) -> str:
        return self.chunk_ids[first_line]

    def distinctive_documents(self, key: str, limit: int = 4) -> list[str]:
        names = []
        for entry in self.checklists[key]["items"]:
            text = entry["text"].lower()
            if text.startswith(GENERIC_ITEMS):
                continue
            record = self.documents.get(_item_key(entry["text"]))
            if record and record["name"] not in names:
                names.append(record["name"])
        return names[:limit]

    def pick(self, group: str, modality: str, offset: int = 0) -> tuple[str, str]:
        found = []
        for code, activity in sorted(self.activities.items()):
            if activity["group"][0] != group:
                continue
            keys = activity["checklists_by_modality"].get(modality, [])
            if keys and keys[0] in self.checklists and activity["name"]:
                found.append((code, keys[0]))
        return found[offset]


def _item_key(text: str) -> str:
    from licenciagraph.sources.documents import item_key

    return item_key(text)


def case(
    builder: Builder,
    case_id: str,
    question: str,
    chunk_ids: list[str],
    facts: list[str],
    concepts: list[str],
    multihop: bool = False,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "question": question,
        "expected_sources": chunk_ids,
        "expected_concepts": concepts,
        "expected_facts": facts,
        "requires_multihop": multihop,
        "evidence": [builder.chunk_text[c] for c in chunk_ids],
    }


def documents_case(builder: Builder, case_id: str, group: str, modality: str, nth=0):
    code, key = builder.pick(group, modality, nth)
    activity = builder.activities[code]
    names = builder.distinctive_documents(key)
    modality_name = MODALITY_QUESTION.get(modality, modality)
    checklist = builder.checklists[key]
    return case(
        builder,
        case_id,
        f"Que documentos específicos a {modality_name} exige para "
        f"{activity['name'].rstrip('.')}?",
        [builder.checklist_chunk(key), builder.activity_chunk(code)],
        [f"O checklist {checklist['code']} exige {name}." for name in names],
        names,
        multihop=True,
    )


def validity_case(builder: Builder, case_id: str, regime: str, offset: int = 0):
    found = [
        (code, a["regimes"][regime])
        for code, a in sorted(builder.activities.items())
        if regime in a["regimes"] and a["regimes"][regime].get("validity")
    ]
    code, info = found[offset]
    activity = builder.activities[code]
    modality = MODALITIES[{"lau": "LAU", "lac": "LAC"}[regime]]
    years = re.sub(r"\D", "", info["validity"])
    return case(
        builder,
        case_id,
        f"Qual a validade e a letra de custo do licenciamento "
        f"({modality}) para {activity['name'][:110].rstrip('.;')}?",
        [builder.activity_chunk(code)],
        [
            f"A validade é de {years} anos.",
            f"A letra de custo é {info['letter']}.",
        ],
        [years, info["letter"]],
    )


def cost_case(builder: Builder, case_id: str, code: str, porte: str, modality: str):
    activity = builder.activities[code]
    row = next(r for r in activity["table"]["rows"] if r["porte"] == porte)
    value = builder.letters[row["letter"]][_cost_column(modality)]
    return case(
        builder,
        case_id,
        f"Quanto custa, em UFR-PB, a análise da {MODALITY_QUESTION[modality]} para "
        f"{activity['name'].rstrip('.')} de porte {PORTE_LABELS[porte].lower()}?",
        [builder.activity_chunk(code), _cost_chunk(builder)],
        [
            f"O porte {PORTE_LABELS[porte].lower()} usa a letra {row['letter']}.",
            f"A análise da {MODALITY_QUESTION[modality]} custa {value} UFR-PB.",
        ],
        [row["letter"], f"{value} UFR-PB"],
        multihop=True,
    )


def porte_case(builder: Builder, case_id: str, code: str, porte: str):
    activity = builder.activities[code]
    row = next(r for r in activity["table"]["rows"] if r["porte"] == porte)
    interval = parse_interval(row["criterion"])
    value = _inside(interval)
    unit = activity["table"]["criterion_label"].split("OBS")[0].strip()
    return case(
        builder,
        case_id,
        f"Qual o porte de {activity['name'].rstrip('.')} com {_format(value)} "
        f"({unit.lower()})?",
        [builder.activity_chunk(code)],
        [
            f"O porte é {PORTE_LABELS[porte].lower()}.",
            f"A letra de custo é {row['letter']}.",
        ],
        [PORTE_LABELS[porte], row["letter"]],
    )


def _inside(interval) -> float:
    low = interval.low if interval.low is not None else 0.0
    if interval.high is None:
        return low + 1
    if interval.low is None:
        return max(interval.high / 2, 1)
    middle = (low + interval.high) / 2
    return round(middle, 1)


def _format(value: float) -> str:
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def _cost_chunk(builder: Builder) -> str:
    return next(
        c
        for first, c in builder.chunk_ids.items()
        if first.startswith("Custos de análise")
    )


def _cost_column(modality: str) -> str:
    return {"LAI": "LAI_LAO", "LAO": "LAI_LAO"}.get(modality, modality)


def sequence_case(builder: Builder, case_id: str):
    chunk = next(
        c
        for first, c in builder.chunk_ids.items()
        if first.startswith("Sequência das licenças")
    )
    return case(
        builder,
        case_id,
        "Em que ordem se obtêm a Licença Prévia, a de Instalação e a de Operação?",
        [chunk],
        [
            "A Licença Prévia vem antes da Licença de Instalação.",
            "A Licença de Instalação vem antes da Licença de Operação.",
        ],
        ["Prévia", "Instalação", "Operação"],
        multihop=True,
    )


def dispensa_case(builder: Builder, case_id: str, offset: int):
    found = [
        (code, a)
        for code, a in sorted(builder.activities.items())
        if "dispensa" in a["regimes"] and a["name"]
    ]
    code, activity = found[offset]
    return case(
        builder,
        case_id,
        f'A atividade "{activity["name"][:100].rstrip(" ;.")}" é passível de dispensa '
        "de licenciamento ambiental?",
        [builder.activity_chunk(code)],
        [
            "A atividade está na lista de atividades passíveis de dispensa de licenciamento."
        ],
        ["dispensa"],
    )


def site_case(
    builder: Builder,
    case_id: str,
    curated: Path,
    doc: str,
    question: str,
    facts,
    concepts,
):
    site = json.loads((curated / "site.json").read_text(encoding="utf-8"))
    document = next(d for d in site["documents"] if d["id"] == doc)
    text = f"{document['title']}. {document['chunks'][0]['text']}"
    chunk_id = f"documento:{doc}:trecho:0001"
    builder.chunk_text[chunk_id] = text
    return case(builder, case_id, question, [chunk_id], facts, concepts)


def negative(case_id: str, question: str, expected: str) -> dict[str, Any]:
    return {"id": case_id, "question": question, "expected_behavior": expected}


def build(curated: Path) -> dict[str, list[dict[str, Any]]]:
    b = Builder(curated)
    dev = [
        documents_case(b, "D01", "16", "LI"),
        documents_case(b, "D02", "32", "LP"),
        documents_case(b, "D03", "40", "LI"),
        documents_case(b, "D04", "48", "LO"),
        documents_case(b, "D05", "56", "LI"),
        validity_case(b, "D06", "lau", 3),
        validity_case(b, "D07", "lac", 2),
        cost_case(b, "D08", "16.15.599", "MEDIO", "LI"),
        cost_case(b, "D09", "08.49.100", "PEQUENO", "LI")
        if False
        else porte_case(b, "D09", "16.15.599", "GRANDE"),
        porte_case(b, "D10", "08.49.100", "PEQUENO"),
        sequence_case(b, "D11"),
        dispensa_case(b, "D12", 4),
        documents_case(b, "D13", "16", "LO"),
        documents_case(b, "D14", "72", "LP"),
        site_case(
            b,
            "D15",
            curated,
            "site-unidades-regionais",
            "Qual o endereço do núcleo regional da SUDEMA em Campina Grande?",
            ["O núcleo regional de Campina Grande fica na Av. Rio Branco, 60, Centro."],
            ["Rio Branco", "60"],
        ),
    ]
    heldout = [
        documents_case(b, "H01", "24", "LP"),
        documents_case(b, "H02", "64", "LI"),
        documents_case(b, "H03", "32", "LO", 2),
        documents_case(b, "H04", "48", "LP", 3),
        documents_case(b, "H05", "56", "LO", 4),
        validity_case(b, "H06", "lau", 10),
        validity_case(b, "H07", "lac", 20),
        cost_case(b, "H08", "16.15.599", "GRANDE", "LO"),
        cost_case(b, "H09", "08.49.100", "GRANDE", "LP")
        if False
        else porte_case(b, "H09", "08.49.100", "MEDIO"),
        porte_case(b, "H10", "16.15.599", "MEDIO"),
        dispensa_case(b, "H11", 30),
        documents_case(b, "H12", "32", "LI", 5),
        documents_case(b, "H13", "48", "LI", 7),
        documents_case(b, "H14", "40", "LO", 3),
        site_case(
            b,
            "H15",
            curated,
            "site-copam",
            "A qual secretaria o COPAM está vinculado atualmente?",
            [
                "O COPAM está vinculado à Secretaria de Estado do Meio Ambiente e Sustentabilidade (SEMAS)."
            ],
            ["SEMAS", "Meio Ambiente e Sustentabilidade"],
        ),
    ]
    dev_negative = [
        negative(
            "DN01",
            "Qual o valor, em reais, da UFR-PB em vigor para converter o custo da licença?",
            "A informação não consta das normas; o valor da UFR-PB em reais não é informado.",
        ),
        negative(
            "DN02",
            "Quais documentos a CETESB exige para licenciar uma fábrica em São Paulo?",
            "Licenciamento de outro estado não consta das normas da SUDEMA.",
        ),
        negative(
            "DN03",
            "A SUDEMA garante a aprovação da licença em 30 dias?",
            "As normas não garantem aprovação em 30 dias; o prazo de análise é outro.",
        ),
        negative(
            "DN04",
            "Que documentos são exigidos para licenciar uma fábrica de foguetes espaciais?",
            "Essa atividade não consta da lista de atividades das normas.",
        ),
    ]
    heldout_negative = [
        negative(
            "HN01",
            "Qual a multa em reais por operar uma avicultura sem licença?",
            "Valores de multa não constam das fontes consultadas.",
        ),
        negative(
            "HN02",
            "O IBAMA emite a Licença de Instalação de uma granja de aves na Paraíba?",
            "As normas tratam do licenciamento estadual pela SUDEMA; licenciamento federal não consta.",
        ),
        negative(
            "HN03",
            "Quanto custa em reais a análise da Licença Prévia da avicultura industrial de porte médio?",
            "O custo consta em UFR-PB; a conversão em reais não consta.",
        ),
        negative(
            "HN04",
            "A Licença de Operação da avicultura industrial dispensa o Cadastro Ambiental Rural?",
            "Premissa falsa ou não confirmada; as normas devem ser consultadas e a premissa não confirmada.",
        ),
        negative(
            "HN05",
            "Qual o telefone do núcleo regional da SUDEMA em Sousa?",
            "Não existe núcleo regional em Sousa nas fontes; há núcleos em Campina Grande e Patos.",
        ),
    ]
    return {
        "questions": dev,
        "heldout_questions": heldout,
        "negative_questions": dev_negative,
        "heldout_negative_questions": heldout_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated", type=Path, default=ROOT / "data/curated")
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, cases in build(args.curated).items():
        target = args.output / f"{name}.json"
        target.write_text(
            json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(name, len(cases))


if __name__ == "__main__":
    main()
