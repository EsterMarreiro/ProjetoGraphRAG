"""Recria o catálogo fictício autoral e o gabarito de avaliação.

As relações são redigidas em prosa; a carga normal deve extraí-las com o LLM,
sem ler esta tabela como um grafo pronto.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COURSES = [
    (
        "01_logica",
        "Lógica de Programação",
        "Fundamentos",
        "Decomposição de problemas",
        "Iniciante",
        "Estudantes",
        20,
        [],
        "Desenvolvimento Web",
    ),
    (
        "02_python",
        "Programação Python",
        "Programação",
        "Implementação de funções",
        "Iniciante",
        "Estudantes",
        30,
        ["Lógica de Programação"],
        "Desenvolvimento Web",
    ),
    (
        "03_estruturas",
        "Estruturas de Dados",
        "Fundamentos",
        "Organização de coleções",
        "Intermediário",
        "Desenvolvedores",
        30,
        ["Programação Python"],
        "Dados e Conhecimento",
    ),
    (
        "04_relacional",
        "Modelagem Relacional",
        "Dados",
        "Projeto de esquemas SQL",
        "Intermediário",
        "Analistas de dados",
        24,
        ["Programação Python"],
        "Dados e Conhecimento",
    ),
    (
        "05_http",
        "Redes e HTTP",
        "Infraestrutura",
        "Compreensão de protocolos",
        "Iniciante",
        "Desenvolvedores",
        18,
        ["Lógica de Programação"],
        "Desenvolvimento Web",
    ),
    (
        "06_apis",
        "APIs Web",
        "Programação",
        "Construção de endpoints",
        "Intermediário",
        "Desenvolvedores",
        28,
        ["Programação Python", "Redes e HTTP"],
        "Desenvolvimento Web",
    ),
    (
        "07_testes",
        "Testes Automatizados",
        "Qualidade",
        "Validação de software",
        "Intermediário",
        "Desenvolvedores",
        24,
        ["APIs Web"],
        "Desenvolvimento Web",
    ),
    (
        "08_conteineres",
        "Contêineres",
        "Infraestrutura",
        "Empacotamento de aplicações",
        "Intermediário",
        "Desenvolvedores",
        20,
        ["APIs Web"],
        "Desenvolvimento Web",
    ),
    (
        "09_integracao",
        "Integração Contínua",
        "Qualidade",
        "Automação de pipelines",
        "Avançado",
        "Desenvolvedores",
        26,
        ["Testes Automatizados", "Contêineres"],
        "Desenvolvimento Web",
    ),
    (
        "10_grafos",
        "Grafos de Conhecimento",
        "Dados",
        "Representação de relações",
        "Avançado",
        "Analistas de dados",
        30,
        ["Modelagem Relacional", "Estruturas de Dados"],
        "Dados e Conhecimento",
    ),
    (
        "11_busca",
        "Busca Semântica",
        "Inteligência Artificial",
        "Recuperação vetorial",
        "Intermediário",
        "Analistas de dados",
        24,
        ["Programação Python", "Estruturas de Dados"],
        "Dados e Conhecimento",
    ),
    (
        "12_hibrida",
        "Recuperação Híbrida",
        "Inteligência Artificial",
        "Fundamentação por evidências",
        "Avançado",
        "Analistas de dados",
        32,
        ["Grafos de Conhecimento", "Busca Semântica"],
        "Dados e Conhecimento",
    ),
]


def main():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    for code, name, area, skill, level, profile, hours, prerequisites, trail in COURSES:
        sentences = [
            f"Título: {name}",
            "Origem: catálogo didático fictício e autoral do projeto TrilhaGraph.",
            "Versão do catálogo: 1.0, 11 de novembro de 2026.",
            f"O curso {name} aborda a área {area} e desenvolve a competência {skill}.",
            f"O curso {name} é classificado no nível {level} "
            f"e é indicado para o perfil {profile}.",
            f"O curso {name} tem código {code} "
            f"e carga horária fictícia de {hours} horas.",
            f"A trilha {trail} inclui o curso {name}.",
        ]
        if prerequisites:
            sentences.extend(
                f"O curso {name} exige como pré-requisito o curso {p}."
                for p in prerequisites
            )
        else:
            sentences.append(f"O curso {name} não exige pré-requisito neste catálogo.")
        sentences.append(
            "As cargas horárias são valores didáticos. Não há "
            "informações sobre preços, inscrições, certificados, "
            "datas de oferta ou equivalência acadêmica."
        )
        (raw / f"{code}.txt").write_text("\n".join(sentences) + "\n", encoding="utf-8")

    cases = [
        (
            (
                "Qual curso iniciante desenvolve Decomposição de "
                "problemas e em qual trilha aparece?"
            ),
            [0],
            ["Lógica de Programação", "Desenvolvimento Web"],
            True,
        ),
        (
            (
                "Qual é o pré-requisito de Programação Python e que "
                "competência esse pré-requisito desenvolve?"
            ),
            [0, 1],
            ["Lógica de Programação", "Decomposição de problemas"],
            True,
        ),
        (
            "Que competência de Python é necessária antes de Estruturas de Dados?",
            [1, 2],
            ["Programação Python", "Implementação de funções"],
            True,
        ),
        (
            (
                "Qual área e competência são abordadas por Modelagem "
                "Relacional e qual curso ela exige?"
            ),
            [3],
            ["Dados", "Projeto de esquemas SQL", "Programação Python"],
            True,
        ),
        (
            (
                "Como Redes e HTTP se conecta a Lógica de Programação e "
                "à Decomposição de problemas?"
            ),
            [0, 4],
            ["Lógica de Programação", "Decomposição de problemas"],
            True,
        ),
        (
            (
                "Quais são os dois pré-requisitos de APIs Web e que "
                "competências eles desenvolvem?"
            ),
            [1, 4, 5],
            [
                "Programação Python",
                "Redes e HTTP",
                "Implementação de funções",
                "Compreensão de protocolos",
            ],
            True,
        ),
        (
            (
                "Que competência é desenvolvida pelo pré-requisito "
                "direto de Testes Automatizados?"
            ),
            [5, 6],
            ["APIs Web", "Construção de endpoints"],
            True,
        ),
        (
            "Que competência de APIs Web é exigida antes de estudar Contêineres?",
            [5, 7],
            ["APIs Web", "Construção de endpoints"],
            True,
        ),
        (
            (
                "Integração Contínua exige quais cursos e quais "
                "competências são desenvolvidas neles?"
            ),
            [6, 7, 8],
            [
                "Testes Automatizados",
                "Contêineres",
                "Validação de software",
                "Empacotamento de aplicações",
            ],
            True,
        ),
        (
            (
                "Quais são os pré-requisitos de Grafos de Conhecimento "
                "e as competências oferecidas por eles?"
            ),
            [2, 3, 9],
            [
                "Modelagem Relacional",
                "Estruturas de Dados",
                "Projeto de esquemas SQL",
                "Organização de coleções",
            ],
            True,
        ),
        (
            (
                "Quais cursos são exigidos antes de Busca Semântica e "
                "em quais áreas esses cursos se enquadram?"
            ),
            [1, 2, 10],
            ["Programação Python", "Estruturas de Dados", "Programação", "Fundamentos"],
            True,
        ),
        (
            (
                "Que competências dos pré-requisitos diretos sustentam "
                "Recuperação Híbrida?"
            ),
            [9, 10, 11],
            [
                "Grafos de Conhecimento",
                "Busca Semântica",
                "Representação de relações",
                "Recuperação vetorial",
            ],
            True,
        ),
        (
            (
                "Qual é a carga horária fictícia de Recuperação Híbrida "
                "e ela comprova equivalência acadêmica?"
            ),
            [11],
            ["32", "equivalência"],
            False,
        ),
        (
            (
                "Qual curso desenvolve Validação de software e qual "
                "desenvolve Automação de pipelines?"
            ),
            [6, 8],
            ["Testes Automatizados", "Integração Contínua"],
            False,
        ),
        (
            (
                "Qual percurso de dois pré-requisitos liga Recuperação "
                "Híbrida a Modelagem Relacional?"
            ),
            [9, 11],
            ["Recuperação Híbrida", "Grafos de Conhecimento", "Modelagem Relacional"],
            True,
        ),
    ]
    output = [
        {
            "id": f"Q{i:02}",
            "question": q,
            "expected_sources": [f"{COURSES[n][0]}.txt" for n in files],
            "expected_concepts": concepts,
            "requires_multihop": multi,
        }
        for i, (q, files, concepts, multi) in enumerate(cases, 1)
    ]
    (ROOT / "data/evaluation/questions.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    (ROOT / "data/evaluation/negative_questions.json").write_text(
        json.dumps(
            [
                {
                    "id": "N01",
                    "question": "Qual é o preço de inscrição em Recuperação Híbrida?",
                    "expected_behavior": (
                        "Informar que o catálogo não informa preços; não "
                        "inventar valor."
                    ),
                },
                {
                    "id": "N02",
                    "question": (
                        "Qual certificado é emitido em APIs Web e em que data "
                        "abrem as inscrições?"
                    ),
                    "expected_behavior": (
                        "Declarar ausência de informações sobre certificados e "
                        "inscrições."
                    ),
                },
                {
                    "id": "N03",
                    "question": (
                        "Qual curso de medicina deste catálogo ensina cardiologia?"
                    ),
                    "expected_behavior": (
                        "Abster-se; o catálogo não contém medicina nem cardiologia."
                    ),
                },
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
