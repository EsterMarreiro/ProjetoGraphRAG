"""Apresentação PDF gerada das evidências reais, com estado parcial explícito."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, help="mantido por compatibilidade")
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
            "TrilhaGraph | Tendências em Ciência da Computação",
            fontsize=10,
            color=(0.35, 0.39, 0.42),
        )
        page.insert_text((894, 514), str(len(document)), fontsize=10)
        return page

    slide(
        "TrilhaGraph\nPercursos de aprendizagem explicáveis",
        f"{authors}\n\nGraphRAG com FalkorDB, API OpenAI e fontes auditáveis.\n"
        "Catálogo didático fictício e autoral: não representa ofertas reais.",
    )
    slide(
        "Problema e objetivo",
        "Uma descrição de curso cita pré-requisitos, mas suas competências "
        "estão em outros documentos.\n\n"
        "Objetivo: conectar essas evidências, explicar a resposta e mostrar "
        "o caminho no grafo.\n\n"
        "Exemplo: qual competência do pré-requisito sustenta um curso avançado?",
    )
    slide(
        "Corpus e LPG",
        "12 cursos fictícios; 2 trilhas; 6 áreas; 3 níveis.\n\n"
        "38 entidades de domínio únicas; 12 documentos; 12 trechos.\n\n"
        "Curso -> PRE_REQUISITO -> Curso -> DESENVOLVE -> Competência\n"
        "Trilha -> INCLUI -> Curso\n\n"
        "Cada relação preserva o trecho de origem. "
        "FalkorDB no lugar do Neo4j, com autorização presencial do professor.",
    )
    page = slide("Quatro pilares implementados", "")
    with pymupdf.open(ROOT / "docs/artigo/arquitetura.pdf") as diagram:
        page.show_pdf_page(pymupdf.Rect(48, 160, 912, 330), diagram, 0)
    page.insert_textbox(
        pymupdf.Rect(52, 365, 908, 470),
        "NER/RE tipadas por LLM; constraints e índices; busca híbrida e dois saltos; "
        "respostas com citações e subgrafo interativo.",
        fontsize=19,
        fontname="helv",
    )
    slide(
        "Grounding e rastreabilidade",
        "O LLM identifica fatos nos TXT e produz respostas estruturadas.\n\n"
        "Cada afirmação declara fontes F e relações G de apoio. "
        "A aplicação valida os IDs e formata as citações.\n\n"
        "Sem informação: abstenção explícita.\n\n"
        "Citação válida não garante correção factual; "
        "a revisão verifica se a pergunta foi atendida.",
    )
    slide(
        "Ingestão: evidência observada",
        "76 relações explícitas esperadas; 76 recuperadas; nenhuma extra.\n\n"
        "25 constraints e grupos de índices dos 8 rótulos em estado operacional.\n\n"
        "Limite: resultado de um catálogo pequeno e controlado. "
        "Não é acurácia em documentos externos.",
    )
    slide(
        "Recuperação em grafo",
        "Entidades citadas na pergunta viram sementes; expansão em 2 saltos.\n\n"
        "Caminho mínimo por pré-requisitos, comunidades de Louvain e PageRank "
        "entram no contexto do LLM.\n\n"
        "Verificador descarta afirmações sem apoio; temas ausentes "
        "(preço, certificado) recebem aviso fixo.",
    )
    slide(
        "Avaliação: 38 casos, juiz com o corpus",
        "Acertos (30 positivas | 8 negativas):\n"
        "Pipeline inicial (gpt-4o-mini): 21 | 6\n"
        "Versão final (gpt-4.1): 30 | 8, sem afirmações falsas\n"
        "Baseline vetorial (gpt-4.1): 18 | 8\n\n"
        "Ablação: recuperação +5, modelo +4, verificador 0.\n"
        "Conjunto retido escrito pelos autores; juiz é um LLM.",
    )
    slide(
        "Demonstração e revisão factual",
        "1. Perguntar pelo curso que desenvolve "
        "Decomposição de problemas e sua trilha.\n"
        "2. Abrir fonte, subgrafo e relações de apoio.\n"
        "3. Perguntar preço ou certificado e verificar a abstenção.\n\n"
        "A primeira versão do juiz marcou como falsas afirmações verdadeiras; "
        "o prompt foi corrigido e conferido com outro modelo.",
    )
    slide(
        "Conclusões e limites",
        "A ingestão reconstruiu o grafo controlado e preservou proveniência.\n\n"
        "A interface permite inspecionar fontes, nós, relações e Cypher.\n\n"
        "Corpus sintético, contexto maior que o do baseline, perguntas retidas "
        "escritas pelos autores e juiz LLM limitam a generalização. "
        "Não se implementou integralmente o framework KAG externo.",
    )
    output = ROOT / "docs/APRESENTACAO.pdf"
    document.save(output, garbage=4, deflate=True)
    print(output)


if __name__ == "__main__":
    main()
