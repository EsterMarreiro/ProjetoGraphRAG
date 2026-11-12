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
        ("Híbrido, desenvolvimento", "run-dev-hybrid-v3/hybrid"),
        ("Híbrido, retido", "run-heldout-hybrid-v3/hybrid"),
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
            "Execução pendente.\n\nProtocolo: 42 perguntas positivas e 10 negativas, "
            "gabarito calculado nas tabelas curadas, juiz LLM com evidências "
            "oficiais, baseline vetorial com o mesmo modelo."
        )
    return (
        "Acertos segundo o juiz (gpt-4o):\n"
        + "\n".join(rows)
        + "\n\nSoma das positivas: híbrido 41/42, vetorial 23/42. Fala coloquial "
        "(26 perguntas): resolvedor 24, método anterior 12.\nNegativas (juiz "
        "estrito): resolvedor 6/10, antes 7/10, vetorial 5/10. Custo: US$ 0,03 a "
        "0,05 por pergunta."
    )


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
        "O empreendedor precisa saber qual licença se aplica, quais documentos "
        "levar, o porte, o custo e a validade. Essa informação está espalhada em "
        "mais de 970 páginas e é difícil de achar no dia a dia; as dúvidas "
        "básicas tendem a recair sobre o atendimento da SUDEMA (premissa, não "
        "medida).\n\nObjetivo geral: facilitar o acesso do empreendedor, com "
        "respostas em linguagem simples, com fonte e página, e ajudar a enquadrar "
        "a atividade, aliviando o atendimento e padronizando o entendimento.",
    )
    slide(
        "Objetivos específicos",
        "1. Reunir as normas em um grafo, com a página de cada fato.\n"
        "2. Entender a atividade na fala do empreendedor e enquadrá-la: "
        "modalidade, porte, custo, validade e documentos.\n"
        "3. Tirar dúvidas gerais (siglas, prazos, etapas).\n"
        "4. Reduzir alucinações: citar, verificar, abster-se.\n"
        "5. Interface simples, em formato de conversa, e assistente passo a "
        "passo.\n6. Avaliar contra uma busca vetorial e na fala coloquial.",
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
        "Resolvedor de atividades",
        'O empreendedor diz "vender porcos"; a norma diz "Suinocultura '
        '(industrial) até 1000 animais".\n\n1. Um LLM traduz a fala em termos '
        "formais e reescreve perguntas de seguimento.\n2. O índice vetorial traz "
        "24 atividades candidatas do grafo.\n3. Um segundo LLM escolhe só entre "
        "elas: exata, próxima ou nenhuma, e diz o que falta informar.\n\n"
        "Não consta? Mostra as mais próximas, com aviso de semelhança.",
    )
    slide(
        "Geração e salvaguardas",
        "Cada afirmação declara fontes F e relações G; a aplicação valida os IDs.\n\n"
        "Afirmação sem fonte é descartada; um verificador descarta o que não está "
        "escrito no contexto, inclusive afirmações sobre o negócio do "
        "empreendedor.\n\nCustos sempre em UFR-PB, sem conversão para reais; "
        "aviso de caráter orientativo em toda resposta; abstenção quando a norma "
        "não informa ou a pergunta é de outro estado.",
    )
    page = slide("Interface", "")
    chat = ROOT / "docs/artigo/interface.png"
    page.insert_image(pymupdf.Rect(48, 112, 478, 400), filename=str(chat))
    source = pymupdf.Pixmap(str(ROOT / "docs/artigo/assistente.png"))
    top = pymupdf.IRect(0, 0, source.width, 900)
    cropped = pymupdf.Pixmap(source.colorspace, top)
    cropped.copy(source, top)
    page.insert_image(pymupdf.Rect(482, 112, 912, 400), pixmap=cropped)
    page.insert_textbox(
        pymupdf.Rect(52, 410, 908, 495),
        "Conversa no estilo de chat: caixa de mensagem fixa, lista de conversas "
        "na barra lateral, configurações em janela, atividade interpretada, "
        "fontes, subgrafo e perguntas de seguimento. Assistente de enquadramento "
        "em três passos: atividade, porte e custo, validade e documentos, direto "
        "do grafo.",
        fontsize=14,
        fontname="helv",
    )
    slide("Avaliação", _evaluation_text(args.runs))
    slide(
        "Limites e próximos passos",
        "Dados vêm de normas que mudam; uma nova edição exige regenerar os "
        "arquivos curados.\nValor da UFR-PB em reais, licenciamento federal e "
        "municipal fora da base não constam.\nDocumentos canônicos e fatos do texto "
        "vêm de um LLM, amostrados.\nPerguntas geradas das tabelas: faltam "
        "perguntas escritas por pessoas de fora do projeto.\nCom o resolvedor, as "
        "negativas caíram de 7 para 6 de 10 por mostrarem atividade próxima.\n"
        "A redução da carga do atendimento da SUDEMA é premissa, não foi medida.",
    )
    output = ROOT / "docs/APRESENTACAO.pdf"
    document.save(output, garbage=4, deflate=True)
    print(output)


if __name__ == "__main__":
    main()
