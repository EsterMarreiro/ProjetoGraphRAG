# RotaGraph Brasil

Sistema GraphRAG/KAG para descoberta de destinos e criação de roteiros turísticos
explicáveis no Brasil.

O projeto transforma documentos de texto em um grafo de conhecimento e combina
busca semântica, busca textual e travessias no grafo para responder perguntas em
linguagem natural. Cada resposta deverá apresentar as fontes e o subgrafo que a
fundamentaram.

## Estado do projeto

- [x] Etapa 1 — proposta, casos de uso e modelagem LPG
- [x] Etapa 2 — infraestrutura e esqueleto da aplicação
- [x] Etapa 3 — elaboração do dataset em arquivos TXT
- [x] Etapa 4 — pipeline Text-to-Graph
- [x] Etapa 5 — recuperação híbrida e algoritmos de grafo
- [x] Etapa 6 — interface e visualização
- [x] Etapa 7 — avaliação, testes e documentação final

## Decisões iniciais

- Domínio: turismo brasileiro.
- Banco de grafos: FalkorDB.
- Linguagem: Python.
- Modelos locais: Ollama, com nomes configuráveis por variáveis de ambiente.
- Interface: Streamlit com visualizacao interativa em Plotly.
- Entradas da ingestão: arquivos `.txt` em português.

O escopo completo da primeira etapa está em
[`docs/01-proposta-modelagem.md`](docs/01-proposta-modelagem.md).

As instruções da infraestrutura estão em
[`docs/02-infraestrutura.md`](docs/02-infraestrutura.md).

A cobertura do corpus e as perguntas de avaliação estão em
[`docs/03-dataset.md`](docs/03-dataset.md).

O pipeline de ingestão está documentado em
[`docs/04-ingestao.md`](docs/04-ingestao.md).

A recuperação híbrida e os algoritmos estão documentados em
[`docs/05-recuperacao.md`](docs/05-recuperacao.md).

A interface Streamlit está documentada em
[`docs/06-interface-streamlit.md`](docs/06-interface-streamlit.md).

A avaliação e o checklist de entrega estão em
[`docs/07-avaliacao-entrega.md`](docs/07-avaliacao-entrega.md). O rascunho do
artigo e o roteiro da defesa estão em [`docs/ARTIGO_TECNICO.md`](docs/ARTIGO_TECNICO.md)
e [`docs/APRESENTACAO.md`](docs/APRESENTACAO.md).

## Interface

```bash
streamlit run streamlit_app.py
```

## Início rápido

```bash
cp .env.example .env
docker compose up -d falkordb
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
rotagraph health
rotagraph setup-schema
rotagraph ingest --data-dir data/raw
streamlit run streamlit_app.py
```

O Ollama deve estar ativo e conter os modelos definidos em `.env`. Por padrão:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

## Avaliação

O conjunto em `data/evaluation/questions.json` possui 15 casos, dos quais 13 são
multi-hop. Após a ingestão, gere os relatórios reproduzíveis:

```bash
rotagraph evaluate \
  --json-output artifacts/evaluation-report.json \
  --markdown-output artifacts/evaluation-report.md
```

## Testes

```bash
python -m unittest discover -s tests -v
python tests/integration_smoke.py
python -m compileall -q src tests
docker compose config -q
```

## Materiais fornecidos

O documento de especificação e o ZIP do professor permanecem na raiz como
referência. O código da aplicação foi desenvolvido separadamente, com estrutura,
documentação e testes próprios.
