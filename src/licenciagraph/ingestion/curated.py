"""Constrói documentos, trechos e grafos a partir dos dados curados."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from licenciagraph.ingestion.canonicalization import slugify
from licenciagraph.ingestion.models import (
    Chunk,
    Document,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from licenciagraph.sources.documents import item_key

MODALITIES = {
    "DISPENSA": "Dispensa de Licenciamento Ambiental",
    "LAC": "Licença Ambiental por Adesão e Compromisso (LAC)",
    "LAU": "Licença Ambiental Única (LAU)",
    "AA": "Autorização Ambiental (AA)",
    "LP": "Licença Prévia (LP)",
    "LI": "Licença de Instalação (LI)",
    "LO": "Licença de Operação (LO)",
    "RLI": "Renovação da Licença de Instalação (RLI)",
    "RLO": "Renovação da Licença de Operação (RLO)",
    "LAI": "Licença de Alteração de Instalação (LAI)",
    "LAO": "Licença de Alteração de Operação (LAO)",
    "LOC": "Licença de Operação Corretiva (LOC)",
    "LOP": "Licença de Operação para Pesquisa Mineral (LOP)",
    "RLOP": "Renovação da Licença de Operação para Pesquisa (RLOP)",
    "LIO": "Licença de Instalação e Operação (LIO)",
    "LTE": "Licença de Transporte Estadual (LTE)",
    "LVPE": "Licença Veicular de Propaganda e Eventos (LVPE)",
    "REGISTRO": "Registro",
    "ENC": "Encerramento",
    "COMPENSACAO": "Termo de compensação ambiental",
}
FAMILIES = {
    "AF": "Atividades florestais",
    "AG": "Agropecuária",
    "AI": "Atividades industriais",
    "CS": "Comércio e serviços",
    "EN": "Geração, transmissão e distribuição de energia",
    "FAU": "Fauna",
    "LAC": "Licenciamento por adesão e compromisso",
    "LAU": "Licença ambiental única",
    "LM": "Lavra de minérios",
    "OC": "Obras civis",
    "RS": "Resíduos e produtos",
    "SIS": "Criação de passeriformes (SISPASS)",
    "SNUC": "Compensação ambiental (SNUC)",
    "SOCIO": "Compensação socioambiental",
    "COMPF": "Compensação florestal",
    "ST": "Sistemas de telecomunicações",
}
PORTES = {
    "MICRO": "Micro",
    "PEQUENO": "Pequeno",
    "MEDIO": "Médio",
    "GRANDE": "Grande",
    "EXTRAORDINARIO": "Extraordinário",
}
REGIME_MODALITY = {
    "dispensa": "DISPENSA",
    "lac": "LAC",
    "lau": "LAU",
    "aa": "AA",
    "florestal": "AA",
    "lte": "LTE",
    "lvpe": "LVPE",
}
COST_MODALITIES = {
    "LP": ["LP"],
    "LI": ["LI"],
    "LO": ["LO"],
    "LOC": ["LOC"],
    "LOP": ["LOP"],
    "LAU": ["LAU"],
    "LAC": ["LAC"],
    "LAI_LAO": ["LAI", "LAO"],
    "AA_LTE": ["AA", "LTE"],
    "LVPE": ["LVPE"],
    "REGISTRO": ["REGISTRO"],
}
STUDY_TOKENS = {
    "RCA": "RCA",
    "RAS": "RAS",
    "PCA": "PCA",
    "EVA": "EVA",
    "EIA": "EIA/RIMA",
    "RIMA": "EIA/RIMA",
}
AGENCY_ALIASES = {"ANAC/Aeronáutica": "ANAC e Aeronáutica", "Outro": ""}


@dataclass(slots=True)
class PlannedChunk:
    chunk: Chunk
    extraction: ExtractionResult | None


@dataclass(slots=True)
class SourceDocument:
    document: Document
    chunks: list[PlannedChunk] = field(default_factory=list)


class GraphBuilder:
    """Acumula entidades e relações de um único trecho."""

    def __init__(self) -> None:
        self.entities: dict[str, ExtractedEntity] = {}
        self.relations: dict[tuple[str, str, str], ExtractedRelationship] = {}

    def entity(self, label: str, key: str, name: str, **props: Any) -> ExtractedEntity:
        ref = f"{label.lower()}:{key}"
        known = self.entities.get(ref)
        if known is None:
            known = ExtractedEntity(
                ref=ref,
                label=label,
                name=name,
                properties={k: v for k, v in props.items() if v not in ("", None)},
                canonical_id=ref,
            )
            self.entities[ref] = known
        return known

    def relate(
        self,
        source: ExtractedEntity,
        relation: str,
        target: ExtractedEntity,
        **props: Any,
    ) -> None:
        self.relations[(source.ref, relation, target.ref)] = ExtractedRelationship(
            source.ref,
            target.ref,
            relation,
            1.0,
            {k: v for k, v in props.items() if v not in ("", None)},
        )

    def result(self) -> ExtractionResult:
        return ExtractionResult(
            list(self.entities.values()), list(self.relations.values())
        )


def modality_entity(builder: GraphBuilder, code: str) -> ExtractedEntity:
    name = MODALITIES.get(code, code)
    return builder.entity("Modalidade", code.lower(), name, sigla=code, descricao=name)


def agency_entity(builder: GraphBuilder, name: str) -> ExtractedEntity | None:
    name = AGENCY_ALIASES.get(name, name)
    if not name:
        return None
    return builder.entity("Orgao", slugify(name), name, sigla=name)


def study_entity(builder: GraphBuilder, name: str) -> ExtractedEntity:
    return builder.entity("Estudo", slugify(name), name)


def _checksum(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def _make_document(
    doc_id: str, title: str, filename: str, planned: list[PlannedChunk]
) -> SourceDocument:
    return SourceDocument(
        Document(
            id=f"documento:{doc_id}",
            title=title,
            path=Path(filename),
            checksum=_checksum([p.chunk.text for p in planned]),
            content="",
        ),
        planned,
    )


def _chunk(doc_id: str, position: int, text: str, page: int | None, kind: str):
    return Chunk(
        id=f"documento:{doc_id}:trecho:{position:04d}",
        document_id=f"documento:{doc_id}",
        position=position,
        text=text,
        page=page,
        kind=kind,
    )


def load_curated(directory: Path) -> dict[str, Any]:
    data = {
        name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("manual", "na101")
    }
    docs = directory / "documentos.json"
    data["documentos"] = (
        json.loads(docs.read_text(encoding="utf-8")) if docs.exists() else {}
    )
    return data


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def checklist_ref(item: dict[str, Any]) -> str:
    return item["key"].lower().replace("@", "-").replace("#", "-")


def checklist_title(item: dict[str, Any]) -> str:
    modality = MODALITIES.get(item["modality"], item["modality"])
    variant = f" ({item['variant']})" if item.get("variant") else ""
    return f"{item['code']} · {modality}{variant}"


def checklist_entity(
    builder: GraphBuilder, item: dict[str, Any], **props: Any
) -> ExtractedEntity:
    return builder.entity(
        "Checklist",
        checklist_ref(item),
        checklist_title(item),
        codigo=item["code"],
        **props,
    )


def checklist_documents(data: dict[str, Any]) -> SourceDocument:
    manual, normalized = data["manual"], data["documentos"]
    planned: list[PlannedChunk] = []
    for position, checklist in enumerate(manual["checklists"], start=1):
        builder = GraphBuilder()
        family = FAMILIES.get(checklist["family"], checklist["family"])
        modality = checklist["modality"]
        modality_name = MODALITIES.get(modality, modality)
        variant = f" ({checklist['variant']})" if checklist["variant"] else ""
        node = checklist_entity(
            builder,
            checklist,
            familia=family,
            modalidade=modality,
            descricao=_clean(" ".join(checklist["heading"])),
            paginas=f"{checklist['pages'][0]}-{checklist['pages'][1]}",
        )
        builder.relate(node, "DO_TIPO", modality_entity(builder, modality))
        lines = []
        for item in checklist["items"]:
            text = _clean(item["text"])
            lines.append(f"{item['number']}. {text[:700]}")
            record = normalized.get(item_key(item["text"]))
            if record is None:
                continue
            document = builder.entity(
                "DocumentoExigido",
                slugify(record["name"])[:80],
                record["name"],
                categoria=record["category"],
                condicao=record["condition"],
            )
            builder.relate(node, "EXIGE_DOCUMENTO", document, ordem=item["number"])
            agency = agency_entity(builder, record["agency"])
            if agency is not None:
                builder.relate(document, "EMITIDO_POR", agency)
            for study in record["studies"]:
                if study != "Outro":
                    builder.relate(node, "EXIGE_ESTUDO", study_entity(builder, study))
        scope = _clean(" ".join(checklist["heading"]))
        text = (
            f"Checklist {checklist['code']}: {modality_name}{variant}. "
            f"Grupo de atividades: {family}. {scope}\n"
            f"Documentos exigidos para abrir o processo (Manual de Licenciamento "
            f"Ambiental da SUDEMA, 5ª edição, p. {checklist['pages'][0]}):\n"
            + "\n".join(lines)
        )
        planned.append(
            PlannedChunk(
                _chunk(
                    "manual-checklists",
                    position,
                    text,
                    checklist["pages"][0],
                    "checklist",
                ),
                builder.result(),
            )
        )
    return _make_document(
        "manual-checklists",
        "Manual de Licenciamento Ambiental, 5ª edição: checklists de documentos",
        "manual-licenciamento-5a-edicao.pdf",
        planned,
    )


def merge_activities(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manual, norm = data["manual"], data["na101"]
    activities: dict[str, dict[str, Any]] = {}

    def record(code: str) -> dict[str, Any]:
        return activities.setdefault(
            code,
            {
                "code": code,
                "name": "",
                "group": ["", ""],
                "subgroup": ["", ""],
                "regimes": {},
                "checklists_by_modality": {},
                "checklists": [],
                "table": None,
                "pages": {},
            },
        )

    for item in norm["master"]:
        entry = record(item["code"])
        entry["name"] = item["name"]
        entry["group"] = [item["group_code"], item["group_name"]]
        entry["subgroup"] = [item["subgroup_code"], item["subgroup_name"]]
        entry["pages"]["na101_lista"] = None
    for item in manual["activities"]:
        entry = record(item["code"])
        entry["name"] = item["name"] or entry["name"]
        if not entry["group"][0]:
            entry["group"] = item["group"]
            entry["subgroup"] = item["subgroup"]
        for regime in item["regime"].split("+"):
            if regime != "licenca":
                entry["regimes"].setdefault(regime, {})
        entry["checklists_by_modality"].update(item["checklists_by_modality"])
        entry["checklists"] = sorted({*entry["checklists"], *item["checklists"]})
        entry["pages"]["manual"] = item["page"]
    for item in norm["listed_activities"]:
        entry = record(item["code"])
        entry["name"] = entry["name"] or item["name"]
        entry["regimes"][item["regime"]] = {
            "validity": item["validity"],
            "letter": item["letter"],
        }
        entry["pages"]["na101_" + item["regime"]] = item["page"]
    listed = {item["code"]: item for item in norm["listed_activities"]}
    for table in norm["porte_tables"]:
        codes = table["codes"]
        if not codes:
            continue
        entry = record(codes[0])
        entry["name"] = entry["name"] or table["name"]
        entry["table"] = table
        entry["pages"]["na101_porte"] = table["pages"][0]
        for alt in table["alt_codes"].values():
            item = listed.get(alt)
            if item and item["regime"] in ("lau", "lac"):
                entry["regimes"][item["regime"]] = {
                    "validity": item["validity"],
                    "letter": item["letter"],
                    "note": item["name"],
                }
    return activities


def _ufr(value: str) -> str:
    return value if value else ""


def _source_label(source: str) -> str:
    return source.replace("na101_", "NA-101 ").replace("manual", "Manual")


def activity_documents(data: dict[str, Any]) -> SourceDocument:
    generic = data["manual"]["generic_checklists"]
    checklists = {c["key"]: c for c in data["manual"]["checklists"]}
    checklist_by_code: dict[str, list[dict[str, Any]]] = {}
    for item in checklists.values():
        checklist_by_code.setdefault(item["code"], []).append(item)
    planned: list[PlannedChunk] = []
    for position, activity in enumerate(
        sorted(merge_activities(data).values(), key=lambda a: a["code"]), start=1
    ):
        if not activity["name"]:
            continue
        builder = GraphBuilder()
        code = activity["code"]
        node = builder.entity(
            "Atividade",
            code,
            activity["name"],
            codigo=code,
            grupo=activity["group"][1],
            subgrupo=activity["subgroup"][1],
            descricao=activity["name"],
        )
        lines = [
            f"Atividade {code}: {activity['name']}.",
        ]
        group_code, group_name = activity["group"]
        subgroup_code, subgroup_name = activity["subgroup"]
        if group_code:
            group = builder.entity(
                "Grupo", group_code, group_name or group_code, codigo=group_code
            )
            lines.append(f"Grupo {group_code} {group_name}.")
            if subgroup_code:
                subgroup = builder.entity(
                    "Subgrupo",
                    subgroup_code,
                    subgroup_name or subgroup_code,
                    codigo=subgroup_code,
                )
                builder.relate(subgroup, "SUBGRUPO_DE", group)
                builder.relate(node, "CLASSIFICADA_EM", subgroup)
                lines[-1] += f" Subgrupo {subgroup_code} {subgroup_name}."
        for regime, info in activity["regimes"].items():
            modality_code = REGIME_MODALITY.get(regime)
            if modality_code is None:
                continue
            modality = modality_entity(builder, modality_code)
            builder.relate(
                node,
                "ENQUADRADA_EM",
                modality,
                validade=info.get("validity", ""),
                letra=info.get("letter", ""),
            )
            sentence = f"Pode ser enquadrada em: {MODALITIES[modality_code]}"
            if info.get("validity"):
                sentence += f", com validade de {info['validity'].lower()}"
            if info.get("letter"):
                sentence += f" e letra de custo {info['letter']}"
            if info.get("note"):
                sentence += f" (caso: {_clean(info['note'])[:160]})"
            if sentence + "." not in lines:
                lines.append(sentence + ".")
        by_modality = dict(activity["checklists_by_modality"])
        if not by_modality and activity["group"][0] == "72":
            by_modality = {m: cs for m, cs in generic["by_modality"].items()}
        for modality_code, keys in sorted(by_modality.items()):
            modality = modality_entity(builder, modality_code)
            resolved = []
            for key in keys:
                match = checklists.get(key) or next(
                    iter(checklist_by_code.get(key, [])), None
                )
                if match is None:
                    continue
                resolved.append(match)
                checklist_node = checklist_entity(builder, match)
                builder.relate(node, "EXIGE_CHECKLIST", checklist_node)
            builder.relate(node, "ENQUADRADA_EM", modality)
            if resolved:
                lines.append(
                    f"{MODALITIES.get(modality_code, modality_code)}: checklist "
                    + " e ".join(c["code"] for c in resolved)
                    + "."
                )
        listed = [
            c
            for k in activity["checklists"]
            if k not in {x for xs in by_modality.values() for x in xs}
            for c in [
                checklists.get(k) or next(iter(checklist_by_code.get(k, [])), None)
            ]
            if c
        ]
        for match in listed:
            checklist_node = checklist_entity(builder, match)
            builder.relate(node, "EXIGE_CHECKLIST", checklist_node)
            builder.relate(
                node, "ENQUADRADA_EM", modality_entity(builder, match["modality"])
            )
            lines.append(
                f"Checklist {match['code']} "
                f"({MODALITIES.get(match['modality'], match['modality'])})."
            )
        table = activity["table"]
        if table and table["rows"]:
            criterion = _clean(table["criterion_label"].split("OBS")[0])
            if criterion:
                node.properties["unidade_porte"] = criterion
            lines.append(
                "Porte (critério: " + (criterion or "não informado") + "), "
                "letra de custo e potencial poluidor (ar, água, solo):"
            )
            for row in table["rows"]:
                porte = builder.entity(
                    "Porte", row["porte"].lower(), PORTES[row["porte"]]
                )
                faixa = builder.entity(
                    "Faixa",
                    f"{code}:{row['porte'].lower()}",
                    f"{activity['name']}, porte {PORTES[row['porte']].lower()}",
                    criterio=row["criterion"] if row["criterion"] != "-" else "",
                    letra=row["letter"],
                    classe=row["letter_class"],
                    ar=row["air"],
                    agua=row["water"],
                    solo=row["soil"],
                    anotacao=row["annotation"],
                    porte=row["porte"],
                )
                builder.relate(node, "TEM_FAIXA", faixa)
                builder.relate(faixa, "DE_PORTE", porte)
                detail = f"- {PORTES[row['porte']]}: {row['criterion'] or 'sem faixa'}"
                if row["letter"]:
                    letter = builder.entity(
                        "LetraCusto",
                        row["letter"].lower(),
                        f"Letra {row['letter']}",
                        sigla=row["letter"],
                    )
                    builder.relate(faixa, "USA_LETRA", letter)
                    detail += f", letra {row['letter']}"
                if row["air"] != "-" and row["air"]:
                    detail += (
                        f", potencial poluidor ar {row['air']}, "
                        f"água {row['water']}, solo {row['soil']}"
                    )
                if row["annotation"]:
                    detail += f", {row['annotation']}"
                    for token in re.findall(r"[A-Z]{2,4}", row["annotation"]):
                        study = STUDY_TOKENS.get(token)
                        if study:
                            builder.relate(
                                faixa, "ESTUDO_DA_FAIXA", study_entity(builder, study)
                            )
                lines.append(detail + ".")
        pages = ", ".join(
            f"{_source_label(source)} p. {page}"
            for source, page in activity["pages"].items()
            if page
        )
        if pages:
            lines.append(f"Fontes: {pages}.")
        planned.append(
            PlannedChunk(
                _chunk(
                    "fichas-atividades",
                    position,
                    "\n".join(lines),
                    next((p for p in activity["pages"].values() if p), None),
                    "atividade",
                ),
                builder.result(),
            )
        )
    return _make_document(
        "fichas-atividades",
        "Fichas de atividades: enquadramento, porte, custo e checklists",
        "na-101-5a-edicao-2026.pdf",
        planned,
    )


def cost_documents(data: dict[str, Any]) -> SourceDocument:
    norm = data["na101"]
    planned: list[PlannedChunk] = []
    builder = GraphBuilder()
    lines = [
        "Custos de análise (NA-101, 5ª edição, Anexo B, p. 19). Cada letra de custo "
        "corresponde a um valor em UFR-PB conforme a modalidade:"
    ]
    for letter, cells in sorted(norm["cost_letters_ufr"].items()):
        node = builder.entity(
            "LetraCusto", letter.lower(), f"Letra {letter}", sigla=letter
        )
        parts = []
        for column, value in cells.items():
            for modality_code in COST_MODALITIES.get(column, []):
                modality = modality_entity(builder, modality_code)
                if re.fullmatch(r"\d+", value):
                    builder.relate(node, "CUSTA", modality, valor_ufr=int(value))
                    parts.append(f"{modality_code} {value} UFR-PB")
                elif value == "ISENTO":
                    builder.relate(node, "CUSTA", modality, valor_ufr=0)
                    parts.append(f"{modality_code} isento")
        lines.append(f"Letra {letter}: " + "; ".join(parts) + ".")
    lines.append(
        "Licença Prévia e Instalação em conjunto somam LP + LI; Licença Prévia, "
        "Instalação e Operação em conjunto somam LP + LI + LO acrescidos de 10%."
    )
    text = "\n".join(lines)
    chunks = _split_text(text, 1800)
    for index, body in enumerate(chunks, start=1):
        planned.append(
            PlannedChunk(
                _chunk("na101-custos", index, body, 19, "custos"),
                builder.result() if index == 1 else None,
            )
        )
    criteria = ["Critérios-base de porte (NA-101, Anexo A, p. 18):"]
    for name, rows in norm["base_criteria"].items():
        criteria.append(f"Critérios-base {name}:")
        for row in rows:
            criteria.append(
                f"- {row['porte'].title()}: área {row['area_m2']} m², faturamento "
                f"{row['faturamento_ufr']} UFR-PB, funcionários {row['funcionarios']}."
            )
    planned.append(
        PlannedChunk(
            _chunk("na101-custos", len(planned) + 1, "\n".join(criteria), 18, "porte"),
            None,
        )
    )
    return _make_document(
        "na101-custos",
        "NA-101, 5ª edição (2026): custos de análise e critérios de porte",
        "na-101-5a-edicao-2026.pdf",
        planned,
    )


FLOW_EDGES = (
    ("LP", "LI"),
    ("LI", "LO"),
    ("LI", "LAI"),
    ("LO", "LAO"),
)
FLOW_TEXT = (
    "Sequência das licenças (Manual de Licenciamento Ambiental, 5ª edição, "
    "capítulo II, p. 20 e 21). A Licença Prévia (LP) é concedida na fase "
    "preliminar do planejamento, aprovando a localização e a concepção e "
    "estabelecendo os requisitos para as próximas fases. A Licença de "
    "Instalação (LI) autoriza a instalação conforme os planos e projetos "
    "aprovados. A Licença de Operação (LO) permite a operação da atividade. "
    "A Licença de Alteração de Instalação (LAI) é condicionada à existência e "
    "à validade da LI, e a Licença de Alteração de Operação (LAO), à da LO. "
    "Assim, LP precede LI, LI precede LO, LI precede LAI e LO precede LAO."
)


def flow_document() -> SourceDocument:
    builder = GraphBuilder()
    for first, second in FLOW_EDGES:
        builder.relate(
            modality_entity(builder, first), "PRECEDE", modality_entity(builder, second)
        )
    planned = [
        PlannedChunk(
            _chunk("fluxo-licencas", 1, FLOW_TEXT, 20, "fluxo"), builder.result()
        )
    ]
    return _make_document(
        "fluxo-licencas",
        "Sequência das licenças ambientais",
        "manual-licenciamento-5a-edicao.pdf",
        planned,
    )


def _split_text(text: str, limit: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for line in text.split("\n"):
        if current and len(current) + len(line) + 1 > limit:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts


def prose_documents(directory: Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    sources = []
    prose = directory / "prosa.json"
    if prose.exists():
        sources.extend(json.loads(prose.read_text(encoding="utf-8"))["documents"])
    site = directory / "site.json"
    if site.exists():
        sources.extend(json.loads(site.read_text(encoding="utf-8"))["documents"])
    for source in sources:
        planned = [
            PlannedChunk(
                _chunk(
                    source["id"],
                    position,
                    f"{source['title']}. {item['text']}",
                    item["page"] or None,
                    "prosa",
                ),
                None,
            )
            for position, item in enumerate(source["chunks"], start=1)
        ]
        filename = source.get("file") or source.get("url", source["id"])
        documents.append(
            _make_document(source["id"], source["title"], filename, planned)
        )
    return documents


def build_structured_documents(curated: Path) -> list[SourceDocument]:
    data = load_curated(curated)
    return [
        checklist_documents(data),
        activity_documents(data),
        cost_documents(data),
        flow_document(),
    ]


def build_documents(curated: Path) -> list[SourceDocument]:
    return [*build_structured_documents(curated), *prose_documents(curated)]
