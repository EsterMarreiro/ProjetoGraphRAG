"""Apresentação PDF gerada das evidências reais, com estado parcial explícito."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def _evaluation_text(runs: Path) -> str:
    rows = []
    for label, name in (
        ("Híbrido, desenvolvimento", "run-dev-hybrid/hybrid"),
        ("Híbrido, retido", "run-heldout-hybrid/hybrid"),
        ("Vetorial, desenvolvimento", "run-dev-vector/vector"),
        ("Vetorial, retido", "run-heldout-vector/vector"),
    ):
        path = runs / f"{name}-judge.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())["summary"]
        if summary.get("total"):
            rows.append(
                f"{label}: {summary['correct']}/{summary['total']}, "
                f"afirmações falsas em {summary['with_false_claims']}"
            )
    if not rows:
        return (
            "Execução pendente: a conta da API ficou sem créditos antes do "
            "experimento.\n\nProtocolo: 30 perguntas positivas e 9 negativas, "
            "gabarito calculado nas tabelas curadas, juiz LLM com evidências "
            "oficiais, baseline vetorial com o mesmo modelo."
        )
    return "Acertos segundo o juiz:\n" + "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--authors-json", type=Path)
    args = parser.parse_args()
    authors = "Autoria em confirmação"
    if args.authors_json:
        authors = ", ".join(json.loads(args.authors_json.read_text())["authors"])
    document = pymupdf.open()

    def slide(title, body):
        page = document.new_page(width=960, height=540)
        page.draw_rect(page.rect, color=None, fill=(0.97, 0.97, 0.95))
        page.insert_textbox(
            pymupdf.Rect(48, 35, 912, 115),
            title,
            fontsize=29,
            fontname="hebo",
            color=(0.08, 0.13, 0.17),
        )
        result = page.insert_textbox(
            pymupdf.Rect(52, 135, 908, 475),
            body,
            fontsize=19,
            fontname="helv",
            lineheight=1.35,
            color=(0.12, 0.17, 0.20),
        )
        if result < 0:
            raise ValueError(f"Texto excedeu slide: {title}")
        page.insert_text(
            (52, 514),
            "LicenciaGraph | Tendências em Ciência da Computação",
            fontsize=10,
            color=(0.35, 0.39, 0.42),
        )
        page.insert_text((894, 514), str(len(document)), fontsize=10)
        return page

    slide(
        "LicenciaGraph",
        f"Guia de licenciamento ambiental da SUDEMA\n{authors}\n\nGraphRAG com FalkorDB e API OpenAI, sobre o Manual de "
        "Licenciamento Ambiental (5ª ed.) e a NA-101 (5ª ed., 2026).\n"
        "Orientativo: não substitui a análise do órgão ambiental.",
    )
    slide(
        "Problema e objetivo",
        "O empreendedor precisa descobrir qual licença se aplica, quais documentos "
        "levar, qual o porte, o custo da análise e a validade.\n\n"
        "A informação está espalhada entre o Manual, a NA-101 e o portal da SUDEMA "
        "e a resposta depende de relações: atividade, modalidade, checklist, porte "
        "e custo.\n\nObjetivo: responder com fonte e página e mostrar o caminho "
        "no grafo.",
    )
    slide(
        "Fontes e dados",
        "Manual (585 páginas): 394 checklists, 4.946 itens, 473 atividades.\n"
        "NA-101 (389 páginas): 614 tabelas de porte, letras de custo em UFR-PB, "
        "validade, dispensa, LAC e LAU.\nPortal da SUDEMA: unidades, COPAM, "
        "legislação.\n\nTabelas: leitores determinísticos com testes de "
        "integridade. Itens dos checklists: LLM, 706 textos em 317 documentos. "
        "Texto corrido: LLM com vocabulário controlado.",
    )
    slide(
        "Grafo de propriedades",
        "Atividade -> ENQUADRADA_EM -> Modalidade\n"
        "Atividade -> EXIGE_CHECKLIST -> Checklist -> EXIGE_DOCUMENTO -> Documento\n"
        "Atividade -> TEM_FAIXA -> Faixa -> USA_LETRA -> Letra -> CUSTA -> Modalidade\n"
        "Modalidade -> PRECEDE -> Modalidade\n\n"
        "Cada relação guarda o trecho e a página de origem. FalkorDB no lugar do "
        "Neo4j, com autorização presencial do professor.",
    )
    page = slide("Arquitetura", "")
    with pymupdf.open(ROOT / "docs/artigo/arquitetura.pdf") as diagram:
        page.show_pdf_page(pymupdf.Rect(48, 160, 912, 330), diagram, 0)
    page.insert_textbox(
        pymupdf.Rect(52, 365, 908, 470),
        "Busca híbrida (vetor, texto e Cypher restrito, fundidos por RRF), "
        "ligação de entidades, expansão em até três saltos, caminho entre "
        "modalidades, Louvain e PageRank, respostas citadas e verificadas.",
        fontsize=19,
        fontname="helv",
    )
    slide(
        "Geração e salvaguardas",
        "Cada afirmação declara fontes F e relações G; a aplicação valida os IDs.\n\n"
        "Afirmação sem fonte é descartada; um verificador descarta o que não está "
        "escrito no contexto.\n\nCustos sempre em UFR-PB, sem conversão para "
        "reais; aviso de caráter orientativo em toda resposta; abstenção quando "
        "a norma não informa.",
    )
    page = slide("Interface", "")
    page.insert_image(
        pymupdf.Rect(60, 110, 320, 490),
        filename=str(ROOT / "docs/artigo/interface.png"),
    )
    page.insert_image(
        pymupdf.Rect(340, 110, 600, 490),
        filename=str(ROOT / "docs/artigo/assistente.png"),
    )
    page.insert_textbox(
        pymupdf.Rect(630, 120, 908, 480),
        "Chat: resposta com fontes, subgrafo interativo e consultas.\n\n"
        "Assistente de enquadramento: atividade, porte, custo em UFR-PB, "
        "validade e checklist, direto do grafo, sem modelo de linguagem.",
        fontsize=15,
        fontname="helv",
    )
    slide("Avaliação", _evaluation_text(args.runs))
    slide(
        "Limites e próximos passos",
        "Dados vêm de normas que mudam; uma nova edição exige regenerar os "
        "arquivos curados.\nValor da UFR-PB em reais, licenciamento federal e "
        "municipal fora da base não constam.\nDocumentos canônicos e fatos do texto "
        "vêm de um LLM, amostrados.\nPerguntas geradas das tabelas: faltam "
        "perguntas escritas por pessoas de fora do projeto.",
    )
    output = ROOT / "docs/APRESENTACAO.pdf"
    document.save(output, garbage=4, deflate=True)
    print(output)


if __name__ == "__main__":
    main()
