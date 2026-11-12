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


TEXT_BY_CHUNK: dict[str, str] = {}


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
        TEXT_BY_CHUNK.update(self.chunk_text)
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

    def distinctive_documents(self, key: str, limit: int = 4) -> list[tuple[str, str]]:
        """(trecho do texto original, termo curto) dos itens específicos."""
        found: list[tuple[str, str]] = []
        for entry in self.checklists[key]["items"]:
            text = re.sub(r"\s+", " ", entry["text"]).strip().rstrip(";.")
            if text.lower().startswith(GENERIC_ITEMS) or len(text) < 12:
                continue
            record = self.documents.get(_item_key(entry["text"]))
            if record is None:
                continue
            short = text[:110].rsplit(" ", 1)[0] if len(text) > 110 else text
            term = " ".join(text.split()[:3]).strip(",;:()")
            if all(short != other for other, _ in found):
                found.append((short, term))
        return found[:limit]

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
    items = builder.distinctive_documents(key)
    modality_name = MODALITY_QUESTION.get(modality, modality)
    checklist = builder.checklists[key]
    return case(
        builder,
        case_id,
        f"Que documentos específicos a {modality_name} exige para "
        f"{activity['name'].rstrip('.')}?",
        [builder.checklist_chunk(key), builder.activity_chunk(code)],
        [f"O checklist {checklist['code']} exige: {text}." for text, _ in items],
        [term for _, term in items],
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
        f"Qual a validade e a letra de custo de enquadramento do licenciamento "
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
        [builder.activity_chunk(code), _cost_chunk(builder, row["letter"])],
        [f"A análise da {MODALITY_QUESTION[modality]} custa {value} UFR-PB."],
        [f"{value} UFR-PB"],
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
        [f"O porte é {PORTE_LABELS[porte].lower()}."],
        [PORTE_LABELS[porte]],
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


def _cost_chunk(builder: Builder, letter: str = "A") -> str:
    return next(
        chunk_id
        for chunk_id, text in builder.chunk_text.items()
        if chunk_id.startswith("documento:na101-custos")
        and f"\nLetra {letter}:" in f"\n{text}"
    )


def _cost_column(modality: str) -> str:
    return {"LAI": "LAI_LAO", "LAO": "LAI_LAO"}.get(modality, modality)


def sequence_case(builder: Builder, case_id: str, variant: bool = False):
    chunk = next(
        c
        for first, c in builder.chunk_ids.items()
        if first.startswith("Sequência das licenças")
    )
    if variant:
        return case(
            builder,
            case_id,
            "Quais licenças de alteração dependem de uma licença anterior válida e de qual?",
            [chunk],
            [
                "A Licença de Alteração de Instalação (LAI) depende da validade da Licença de Instalação (LI).",
                "A Licença de Alteração de Operação (LAO) depende da validade da Licença de Operação (LO).",
            ],
            ["LAI", "LAO"],
            multihop=True,
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


def prose_text(curated: Path, *needles: str) -> list[str]:
    found = []
    prose = json.loads((curated / "prosa.json").read_text(encoding="utf-8"))
    for document in prose["documents"]:
        for chunk in document["chunks"]:
            if any(needle in chunk["text"] for needle in needles):
                found.append(f"{document['title']}. {chunk['text']}")
    return found


def site_text(curated: Path, doc: str) -> list[str]:
    site = json.loads((curated / "site.json").read_text(encoding="utf-8"))
    document = next(d for d in site["documents"] if d["id"] == doc)
    return [f"{document['title']}. {c['text']}" for c in document["chunks"]]


def negative(
    case_id: str,
    question: str,
    expected: str,
    chunks: list[str] | None = None,
    texts: list[str] | None = None,
) -> dict[str, Any]:
    entry = {"id": case_id, "question": question, "expected_behavior": expected}
    evidence = [TEXT_BY_CHUNK[c] for c in chunks or []] + list(texts or [])
    if evidence:
        entry["evidence"] = evidence
    return entry


def usable_bands(builder: Builder) -> dict[str, list[tuple[str, str]]]:
    """(atividade, porte) com faixa numérica e letra de custo, por grupo."""
    by_group: dict[str, list[tuple[str, str]]] = {}
    for code, activity in sorted(builder.activities.items()):
        table = activity["table"]
        if not table or not activity["name"]:
            continue
        for row in table["rows"]:
            if (
                row["letter"]
                and row["porte"] in PORTE_LABELS
                and parse_interval(row["criterion"])
            ):
                by_group.setdefault(code[:2], []).append((code, row["porte"]))
    return by_group


def round_robin(groups: dict[str, list], start: int, count: int) -> list:
    """Intercala os grupos para a amostra cobrir áreas diferentes."""
    order = sorted(groups)
    picked, round_ = [], start
    while len(picked) < count and round_ < start + 50:
        for group in order:
            items = groups[group]
            if round_ < len(items) and len(picked) < count:
                picked.append(items[round_])
        round_ += 1
    return picked


def valid_documents(builder: Builder, group: str, modality: str, nth: int):
    found = []
    for offset in range(80):
        try:
            _, key = builder.pick(group, modality, offset)
        except IndexError:
            break
        if len(builder.distinctive_documents(key)) >= 3:
            found.append(offset)
    if not found:
        raise IndexError(f"sem checklist com 3 documentos em {group}/{modality}")
    return found[min(nth, len(found) - 1)]


DOC_PLAN = [
    ("16", "LI"), ("32", "LP"), ("40", "LI"), ("48", "LO"),
    ("56", "LI"), ("64", "LP"), ("72", "LI"), ("64", "LO"),
    ("32", "LO"), ("48", "LP"), ("56", "LO"), ("40", "LO"),
]  # fmt: skip
HELDOUT_DOC_PLAN = [
    ("16", "LO"), ("32", "LI"), ("40", "LP"), ("48", "LI"),
    ("56", "LP"), ("64", "LI"), ("72", "LO"), ("64", "LAI"),
    ("32", "LOC"), ("48", "RLO"), ("56", "RLO"), ("40", "LOC"),
]  # fmt: skip


def split(builder: Builder, prefix: str, held: bool) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    plan = HELDOUT_DOC_PLAN if held else DOC_PLAN
    nth = 1 if held else 0
    for group, modality in plan[:7]:
        offset = valid_documents(builder, group, modality, nth)
        cases.append(
            documents_case(
                builder, f"{prefix}{len(cases) + 1:02d}", group, modality, offset
            )
        )
    lau = 8 if not held else 22
    lac = 5 if not held else 30
    cases.append(validity_case(builder, f"{prefix}{len(cases) + 1:02d}", "lau", lau))
    for k in range(2):
        cases.append(
            validity_case(builder, f"{prefix}{len(cases) + 1:02d}", "lac", lac + 6 * k)
        )
    bands = usable_bands(builder)
    start = 1 if held else 0
    modalities = ["LP", "LI", "LO", "LI", "LO"]
    for k, (code, porte) in enumerate(round_robin(bands, start, 4)):
        modality = modalities[k % len(modalities)]
        cases.append(
            cost_case(builder, f"{prefix}{len(cases) + 1:02d}", code, porte, modality)
        )
    for code, porte in round_robin(bands, start + 6, 3):
        cases.append(porte_case(builder, f"{prefix}{len(cases) + 1:02d}", code, porte))
    base = 4 if not held else 40
    for k in range(2):
        cases.append(
            dispensa_case(builder, f"{prefix}{len(cases) + 1:02d}", base + 11 * k)
        )
    cases.append(sequence_case(builder, f"{prefix}{len(cases) + 1:02d}", variant=held))
    return cases


def build(curated: Path) -> dict[str, list[dict[str, Any]]]:
    b = Builder(curated)
    dev = split(b, "D", held=False)
    heldout = split(b, "H", held=True)
    dev.append(
        site_case(
            b, f"D{len(dev) + 1:02d}", curated, "site-unidades-regionais",
            "Qual o endereço do núcleo regional da SUDEMA em Campina Grande?",
            ["O núcleo regional de Campina Grande fica na Av. Rio Branco, 60, Centro."],
            ["Rio Branco", "60"],
        )
    )  # fmt: skip
    heldout.append(
        site_case(
            b, f"H{len(heldout) + 1:02d}", curated, "site-copam",
            "A qual secretaria o COPAM está vinculado atualmente?",
            ["O COPAM está vinculado à Secretaria de Estado do Meio Ambiente e Sustentabilidade (SEMAS)."],
            ["SEMAS", "Meio Ambiente e Sustentabilidade"],
        )
    )  # fmt: skip
    cost_chunk = _cost_chunk(b, "H")
    dev_negative = [
        negative("DN01", "Qual o valor, em reais, da UFR-PB em vigor para converter o custo da licença?", "A informação não consta das normas; o valor da UFR-PB em reais não é informado."),
        negative("DN02", "Quais documentos a CETESB exige para licenciar uma fábrica em São Paulo?", "Licenciamento de outro estado não consta das normas da SUDEMA."),
        negative("DN03", "A SUDEMA garante a aprovação da licença em 30 dias?", "As normas não garantem aprovação em 30 dias; os prazos de análise são de 3 a 6 meses conforme a licença.", None, prose_text(
            curated, "6 (seis) meses", "prazo de até 06 (seis) meses", "trinta (30) dias úteis"
        )),
        negative("DN04", "Que documentos são exigidos para licenciar uma fábrica de foguetes espaciais?", "Essa atividade não consta da lista de atividades das normas."),
        negative("DN05", "Posso pagar a taxa de análise da licença em 12 parcelas?", "O parcelamento da taxa não consta das normas consultadas."),
    ]  # fmt: skip
    heldout_negative = [
        negative("HN01", "Qual a multa em reais por operar uma avicultura sem licença?", "Valores de multa não constam das fontes consultadas."),
        negative("HN02", "O IBAMA emite a Licença de Instalação de uma granja de aves na Paraíba?", "As normas tratam do licenciamento estadual pela SUDEMA; licenciamento federal não consta.", None, prose_text(curated, "licenciamento ambiental na SUDEMA")),
        negative("HN03", "Quanto custa em reais a análise da Licença Prévia da avicultura industrial de porte médio?", "O custo consta em UFR-PB; a conversão em reais não consta.", [cost_chunk]),
        negative("HN04", "Qual o telefone do núcleo regional da SUDEMA em Sousa?", "Não existe núcleo regional em Sousa nas fontes; há núcleos em Campina Grande e Patos.", None, site_text(curated, "site-unidades-regionais")),
        negative("HN05", "Que documentos a SUDEMA exige para licenciar uma plataforma de petróleo offshore?", "Essa atividade não consta da lista de atividades das normas."),
    ]  # fmt: skip
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
