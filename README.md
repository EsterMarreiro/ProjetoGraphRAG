# LicenciaGraph

Assistente de licenciamento ambiental da SUDEMA (Paraíba) com GraphRAG. O empreendedor pergunta em linguagem natural (que licença preciso, quais documentos levo, qual o porte, quanto custa a análise em UFR-PB, quanto tempo vale) e recebe uma resposta com **fonte e página**, o **subgrafo usado** e as **consultas executadas**. Uma segunda aba, o **assistente de enquadramento**, consulta o grafo diretamente, sem modelo de linguagem: atividade, porte, custo, validade e checklist de documentos.

## Objetivo

**Objetivo geral.** Facilitar o acesso do empreendedor paraibano à informação de licenciamento ambiental da SUDEMA, hoje espalhada entre o Manual de Licenciamento Ambiental (585 páginas), a NA-101 (389 páginas) e o portal, e difícil de achar e de interpretar no dia a dia. A aplicação responde em linguagem simples, com fonte e página, e ajuda a enquadrar a atividade do empreendedor.

**Por que importa.** Quem precisa de uma licença hoje procura as regras em documentos extensos ou recorre ao atendimento da SUDEMA, onde as mesmas dúvidas básicas se repetem. A premissa do projeto é que uma ferramenta de apoio pode aliviar essa demanda e padronizar o entendimento de todos (empreendedores, consultores e o próprio atendimento). Essa premissa **não foi medida**: a avaliação mede a qualidade das respostas, não a redução da carga do atendimento.

**Objetivos específicos.**

1. Reunir as normas em um grafo de conhecimento, com a página de origem de cada fato.
2. Entender a atividade descrita com as palavras do empreendedor ("vender porcos", "estúdio de tatuagem"), enquadrá-la entre as atividades listadas e dizer modalidade, porte, custo em UFR-PB, validade e documentos, pedindo os dados que faltam.
3. Tirar dúvidas gerais (o que é LP, LI, LO, LAC, LAU, prazos, etapas) em linguagem acessível.
4. Reduzir alucinações: citar fonte e página, verificar as afirmações, abster-se quando a norma não informa e nunca afirmar valores sem fonte.
5. Oferecer uma interface simples, no formato de conversa, com um assistente de enquadramento passo a passo.
6. Avaliar o sistema contra uma busca vetorial simples e medir se ele identifica a atividade em fala coloquial.

A aplicação é **orientativa**. A fonte oficial é a SUDEMA; as respostas não substituem a análise técnica do órgão ambiental nem a legislação vigente.

## Fontes

| Fonte | Conteúdo usado |
|---|---|
| Manual de Licenciamento Ambiental, 5ª edição (SUDEMA) | capítulos conceituais; Anexo A: atividades por modalidade e 394 checklists de documentos |
| Norma Administrativa NA-101, 5ª edição (2026) | porte, potencial poluidor, letras de custo em UFR-PB, validade, dispensa, LAC, LAU e observações de custos |
| Portal da SUDEMA | unidades regionais, COPAM, licenciamento municipal, legislação, EIA/RIMA, fiscalização |

Os PDFs não são versionados: o Manual vem do portal da SUDEMA e a NA-101 de `sudema.pb.gov.br/institucional/copam-1/normas-administrativas`; coloque-os em `data/sources/` como `manual-licenciamento-5a-edicao.pdf` e `na-101-5a-edicao-2026.pdf`. O que o projeto extrai deles fica em `data/curated/` (versionado, com a página de origem de cada fato).

## Como funciona

1. **Tabelas** (Anexo A e NA-101): leitores determinísticos com PyMuPDF (`src/licenciagraph/sources/`). A modalidade de cada checklist vem do título da própria página, não da coluna da tabela. 394 checklists, 4.946 itens, 825 atividades, 614 tabelas de porte e a matriz de custos, verificados por testes de integridade cruzada.
2. **Itens dos checklists**: um LLM normaliza os 706 textos distintos em 317 documentos canônicos, com categoria e órgão emissor (`scripts/normalize_documents.py`, resultado em `data/curated/documentos.json`).
3. **Texto corrido** (capítulos, NA-101, portal): um LLM extrai fatos institucionais (órgão emite modalidade, órgão opera sistema, vínculo entre órgãos) com vocabulário controlado; o que foge do vocabulário é descartado.
4. **Grafo** no FalkorDB: `Atividade`, `Grupo`, `Subgrupo`, `Modalidade`, `Checklist`, `DocumentoExigido`, `Estudo`, `Orgao`, `Sistema`, `Norma`, `Porte`, `Faixa`, `LetraCusto`, com proveniência por trecho (`Documento` e `Trecho`).
5. **Resolvedor de atividades**: traduz a fala do empreendedor ("vender porcos", "estúdio de tatuagem") em termos formais, busca candidatas no grafo e um LLM escolhe só entre elas (exata, próxima ou nenhuma); atividade que não consta recebe as mais próximas com aviso de semelhança e a lista do que falta informar. Também entende perguntas de seguimento.
6. **Recuperação híbrida**: vetor, texto e Cypher planejado por LLM em gramática finita, fundidos por RRF; ligação de entidades pela pergunta (por nome, sigla e vetor de atividades); expansão em até três saltos pelas arestas de saída, restrita à modalidade e ao porte citados; caminho mais curto entre modalidades (`PRECEDE`), comunidades de Louvain entre atividades que compartilham checklists e PageRank.
7. **Geração** com afirmações citadas, verificador, regra de custo em UFR-PB (sem conversão para reais) e aviso de caráter orientativo.

## Reprodução local

Requisitos: Python 3.12 ou superior, Docker e uma chave da API da OpenAI.

```bash
cp .env.example .env   # preencha OPENAI_API_KEY; o .env nunca é versionado
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
docker compose up -d falkordb
licenciagraph health
licenciagraph ingest --reset
streamlit run streamlit_app.py
```

`licenciagraph ingest` lê apenas `data/curated/`; os PDFs só são necessários para regenerar esses arquivos (`python scripts/build_curated.py`, `python scripts/fetch_site.py`). Modelos padrão: `gpt-4.1` (plano, resposta, verificação e normalização), `gpt-4o-mini` (fatos do texto) e `text-embedding-3-small` com 768 dimensões. A ingestão completa custa poucos dólares.

```bash
licenciagraph ask 'Quais documentos preciso para a Licença de Instalação de uma avicultura industrial?'
python -m pytest
ruff check src tests scripts
```

## Organização

- `src/licenciagraph/sources/`: leitores das fontes oficiais e normalização dos documentos.
- `src/licenciagraph/ingestion/`: construção dos trechos e do grafo, extração de fatos, escrita no FalkorDB.
- `src/licenciagraph/retrieval/`: busca híbrida, expansão, algoritmos de grafo, geração e verificação.
- `src/licenciagraph/guide.py`: assistente de enquadramento (consultas diretas ao grafo).
- `src/licenciagraph/ui/`: interface Streamlit no formato de chat (`theme.py` com o tema azul e verde; `assets/` com o símbolo e os ícones copiados do repositório da SUDEMA); `src/licenciagraph/evaluation/`: avaliador e juiz.
- `scripts/`: geração dos dados curados, dos conjuntos de avaliação e do experimento.
- `data/curated/`: dados extraídos; `data/evaluation/`: perguntas com gabarito.
- `docs/`: documentação por etapa (modelagem, infraestrutura, fontes, ingestão, recuperação, interface, avaliação) e auditoria dos requisitos.

## Estado

Produto e dados prontos e testados. Avaliação (42 positivas, 10 negativas, 26 perguntas em fala coloquial), artigo, slides e guias prontos em `docs/`; pendente o envio ao GitHub; a equipe consta em `docs/artigo/equipe.json`. Resultado final, com o resolvedor: 41/42 nas positivas (baseline vetorial 23/42), 24/26 na fala coloquial (antes 12/26) e 6/10 nas negativas (antes 7/10; vetorial 5/10).

## Limites

O Manual e a NA-101 mudam; os testes de integridade permitem reingerir uma nova edição. Licenciamento federal, municipal fora da base e de outros estados não consta. O valor da UFR-PB em reais não consta das fontes. Documentos canônicos e fatos do texto vêm de um LLM e foram amostrados, não revisados um a um. A avaliação usa perguntas geradas a partir das tabelas curadas, de modo que sua cobertura é limitada; perguntas escritas por quem não conhece o sistema devem complementá-la.
