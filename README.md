# TrilhaGraph

Assistente GraphRAG/KAG para explorar percursos de aprendizagem em engenharia de software. Responde a partir de um **catálogo didático fictício e autoral** de 12 cursos; não representa ofertas reais, certificados ou equivalência acadêmica.

Combina embeddings, busca textual e Cypher selecionado por LLM em gramática restrita. A travessia de dois saltos conecta cursos, pré-requisitos, competências, áreas, públicos e trilhas. O chat exibe fontes, subgrafo interativo, nós, consultas e confiança heurística.

## Reprodução local

Requisitos: Python 3.12 ou superior, Docker e Compose. Execute na raiz:

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
docker compose up -d falkordb ollama
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen2.5:3b
trilhagraph health
trilhagraph setup-schema
trilhagraph ingest --data-dir data/raw
streamlit run streamlit_app.py
```

O modelo de chat padrão é `qwen2.5:3b`; pode ser trocado em `CHAT_MODEL`. Tempo e memória de inferência dependem do hardware. A API Ollama deve estar em `http://localhost:11434`. O `.env.example` usa FalkorDB em `6380` e navegador em `3001`, evitando conflito com Redis na porta 6379. Use as mesmas variáveis ao subir Compose e executar Python.

```bash
trilhagraph ask 'Que competências dos pré-requisitos sustentam Recuperação Híbrida?'
trilhagraph evaluate --json-output artifacts/evaluation-report.json --markdown-output artifacts/evaluation-report.md
python -m unittest discover -s tests -v
python tests/integration_smoke.py
ruff check src tests scripts
```

O smoke test usa FalkorDB real em um grafo temporário próprio, com extrator e embeddings determinísticos. Não comprova a qualidade do LLM. A avaliação real usa os modelos configurados, sem respostas pré-gravadas. Perguntas negativas para revisão factual estão em `data/evaluation/negative_questions.json`.

## Organização e documentação

- [Proposta e esquema LPG](docs/01-proposta-modelagem.md).
- [Infraestrutura](docs/02-infraestrutura.md), [corpus](docs/03-dataset.md), [ingestão](docs/04-ingestao.md), [recuperação](docs/05-recuperacao.md) e [interface](docs/06-interface-streamlit.md).
- [Auditoria dos requisitos](docs/AUDITORIA_REQUISITOS.md), [avaliação e entrega](docs/07-avaliacao-entrega.md).
- [Texto do artigo](docs/ARTIGO_TECNICO.md), [PDF SBC](docs/artigo/main.pdf), [fontes LaTeX](docs/artigo/) e [apresentação PDF](docs/APRESENTACAO.pdf), ainda em fechamento com resultados completos/autoria.
- [Roteiro da defesa](docs/APRESENTACAO.md) e [evidências de execução](artifacts/README.md).

A implementação é independente do ZIP do professor. O novo domínio substitui o turismo da versão anterior; o namespace `trilhagraph_aprendizagem` mantém o grafo separado. Os materiais da Aula 08 e os modelos de TCC foram analisados e estão identificados em [referências da atividade](docs/referencias/README.md). FalkorDB substitui Neo4j conforme autorização do professor informada pelo usuário.

O catálogo pode ser recriado com `python scripts/build_catalog.py`. Essa rotina escreve os TXT e gabaritos; o pipeline normal extrai entidades e relações da prosa via LLM. A tabela do gerador não é carregada diretamente no grafo.

## Estado da entrega

A troca de tema e corpus foi implementada. A execução integral com LLM, a revisão factual, a verificação visual e o artigo SBC/PDF final ainda estão em andamento. Consulte a auditoria para separar implementação de evidência empírica.
