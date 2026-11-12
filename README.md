# TrilhaGraph

Assistente GraphRAG/KAG para explorar percursos de aprendizagem em engenharia de software. Responde a partir de um **catálogo didático fictício e autoral** de 12 cursos; não representa ofertas reais, certificados ou equivalência acadêmica.

Combina embeddings OpenAI, busca textual e Cypher selecionado por LLM em gramática restrita. Entidades citadas na pergunta viram sementes de uma expansão em dois saltos, com caminho mínimo por pré-requisitos, comunidades de Louvain e PageRank; as afirmações são verificadas antes de serem exibidas. O chat exibe fontes, subgrafo interativo, nós, consultas e confiança heurística.

## Reprodução local

Requisitos: Python 3.12 ou superior, Docker e Compose e uma chave da API da OpenAI. Execute na raiz:

```bash
cp .env.example .env   # preencha OPENAI_API_KEY; o .env nunca é versionado
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
docker compose up -d falkordb
trilhagraph health
trilhagraph setup-schema
trilhagraph ingest --data-dir data/raw
streamlit run streamlit_app.py
```

O modelo de chat padrão é `gpt-4.1` (planejamento, geração e verificação), o de extração é `gpt-4o-mini` (`EXTRACTION_MODEL`) e o de embeddings é `text-embedding-3-small` com 768 dimensões; todos são configuráveis. `VERIFIER_MODEL` vazio desliga a verificação. Mudar modelo de embeddings ou dimensão exige novo `FALKORDB_GRAPH` e nova ingestão. O `.env.example` usa FalkorDB em `6380` e navegador em `3001`, evitando conflito com Redis na porta 6379. A chave pode ser gerada em https://platform.openai.com/api-keys; o custo de uma ingestão e avaliação completas do catálogo é de poucos centavos de dólar.

```bash
trilhagraph ask 'Que competências dos pré-requisitos sustentam Recuperação Híbrida?'
trilhagraph evaluate --json-output artifacts/evaluation-report.json --markdown-output artifacts/evaluation-report.md
python -m unittest discover -s tests -v
python tests/integration_smoke.py
ruff check src tests scripts
```

O smoke test usa FalkorDB real em um grafo temporário próprio, com extrator e embeddings determinísticos. Não comprova a qualidade do LLM. A avaliação real usa os modelos configurados, sem respostas pré-gravadas; `python scripts/run_experiment.py` ingere e executa os modos híbrido e vetorial, registrando tudo em `artifacts/run-*`. Perguntas negativas para revisão factual estão em `data/evaluation/negative_questions.json`.

## Organização e documentação

- [Proposta e esquema LPG](docs/01-proposta-modelagem.md).
- [Infraestrutura](docs/02-infraestrutura.md), [corpus](docs/03-dataset.md), [ingestão](docs/04-ingestao.md), [recuperação](docs/05-recuperacao.md) e [interface](docs/06-interface-streamlit.md).
- [Auditoria dos requisitos](docs/AUDITORIA_REQUISITOS.md), [avaliação e entrega](docs/07-avaliacao-entrega.md).
- [Texto do artigo](docs/ARTIGO_TECNICO.md), [PDF SBC](docs/artigo/main.pdf), [fontes LaTeX](docs/artigo/) e [apresentação PDF](docs/APRESENTACAO.pdf); falta apenas a autoria.
- [Roteiro da defesa](docs/APRESENTACAO.md) e [evidências de execução](artifacts/README.md).
- Guias para a apresentação: [completo](docs/guia/TrilhaGraph-guia-completo.pdf) e [resumido](docs/guia/TrilhaGraph-guia-resumo.pdf), com fontes LaTeX em `docs/guia/`.

A implementação é independente do ZIP do professor. O novo domínio substitui o turismo da versão anterior; o namespace `trilhagraph_aprendizagem` mantém o grafo separado. Os materiais da Aula 08 e os modelos de TCC foram analisados e estão identificados em [referências da atividade](docs/referencias/README.md). FalkorDB substitui Neo4j (a atividade cita Neo4j) com autorização presencial do professor, informada pela equipe.

O catálogo pode ser recriado com `python scripts/build_catalog.py`. Essa rotina escreve os TXT e gabaritos; o pipeline normal extrai entidades e relações da prosa via LLM. A tabela do gerador não é carregada diretamente no grafo.

## Resultados

Julgados por um LLM com acesso ao corpus (`gpt-4o`) sobre 30 perguntas positivas (15 de desenvolvimento e 15 retidas) e 8 negativas:

| Configuração | Positivas corretas | Negativas |
|---|---:|---:|
| Pipeline inicial (gpt-4o-mini) | 21/30 | 6/8 |
| Versão final (gpt-4.1) | 30/30 | 8/8 |
| Baseline vetorial (gpt-4.1) | 18/30 | 8/8 |

A extração reproduziu 76 de 76 relações esperadas. A ablação mostra que o ganho vem da recuperação e do modelo, e que o verificador não mudou o resultado com `gpt-4.1`. Limites: corpus pequeno e sintético, perguntas retidas escritas pelos próprios autores e juiz LLM. Registros em `artifacts/README.md`; análise em `docs/ARTIGO_TECNICO.md` e `docs/AUDITORIA_REQUISITOS.md`.

## Pendências fora do código

Nomes da equipe (2 a 5 alunos) para o artigo e a defesa; acesso do docente ao repositório.
