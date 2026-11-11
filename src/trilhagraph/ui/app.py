"""Aplicacao Streamlit do TrilhaGraph."""

from __future__ import annotations

import streamlit as st

from trilhagraph.retrieval.generation import SCOPE_CAVEAT
from trilhagraph.retrieval.models import GroundedAnswer
from trilhagraph.ui.filters import LearningFilters
from trilhagraph.ui.graph_view import LABEL_COLORS, create_graph_figure
from trilhagraph.ui.runtime import UIRuntime, build_runtime

AREAS = (
    "",
    "Fundamentos",
    "Programação",
    "Dados",
    "Infraestrutura",
    "Qualidade",
    "Inteligência Artificial",
)
COMPETENCIES = (
    "",
    "Decomposição de problemas",
    "Implementação de funções",
    "Construção de endpoints",
    "Recuperação vetorial",
    "Fundamentação por evidências",
)
PROFILES = ("", "Estudantes", "Desenvolvedores", "Analistas de dados")
LEVELS = ("", "Iniciante", "Intermediário", "Avançado")


@st.cache_resource(
    scope="session",
    show_spinner="Conectando aos serviços...",
    on_release=lambda runtime: runtime.close(),
)
def get_runtime() -> UIRuntime:
    return build_runtime()


def main() -> None:
    st.set_page_config(
        page_title="TrilhaGraph",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("📚 TrilhaGraph")
    st.caption("Percursos de aprendizagem explicados com evidências e grafos")

    try:
        runtime = get_runtime()
    except Exception as exc:
        st.error(f"Não foi possível conectar ao FalkorDB: {exc}")
        st.info("Inicie a infraestrutura com `docker compose up -d falkordb`.")
        st.stop()

    filters, services_ready = render_sidebar(runtime)
    initialize_history()
    render_history()

    prompt = st.chat_input(
        "Pergunte sobre cursos, competências, trilhas ou pré-requisitos...",
        disabled=not services_ready,
    )
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        augmented_question = filters.augment_question(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Buscando evidências e percorrendo o grafo..."):
                    answer = runtime.service.answer(augmented_question)
                render_answer(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer.answer, "answer": answer}
                )
            except Exception as exc:
                message = f"Não foi possível responder: {exc}"
                st.error(message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": message}
                )


def render_sidebar(runtime: UIRuntime) -> tuple[LearningFilters, bool]:
    with st.sidebar:
        st.header("Preferências de aprendizagem")
        area = st.selectbox("Área", AREAS)
        competency = st.selectbox("Competência", COMPETENCIES)
        profile = st.selectbox("Perfil", PROFILES)
        level = st.selectbox("Nível", LEVELS)

        st.divider()
        st.subheader("Estado dos serviços")
        database_ok = _database_status(runtime)
        llm_ok = _llm_status(runtime)

        st.divider()
        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption(
            "Catálogo didático fictício. As trilhas não representam ofertas reais "
            "nem equivalência acadêmica."
        )

    return (
        LearningFilters(area, competency, profile, level),
        database_ok and llm_ok,
    )


def _database_status(runtime: UIRuntime) -> bool:
    try:
        runtime.connection.healthcheck()
        st.success(
            f"FalkorDB conectado · `{runtime.settings.falkordb_graph}`",
            icon="✅",
        )
        return True
    except Exception as exc:
        st.error(f"FalkorDB indisponível: {exc}", icon="🚨")
        return False


def _llm_status(runtime: UIRuntime) -> bool:
    try:
        models = runtime.llm.list_models()
    except Exception as exc:
        st.error(f"OpenAI indisponível: {exc}", icon="🚨")
        return False
    missing = [
        model
        for model in (runtime.settings.chat_model, runtime.settings.embedding_model)
        if model not in models
    ]
    if missing:
        st.warning(f"Modelos indisponíveis na OpenAI: {', '.join(missing)}", icon="⚠️")
        return False
    st.success("OpenAI e modelos disponíveis", icon="✅")
    return True


def initialize_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_history() -> None:
    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("**Experimente perguntar:**")
            st.markdown(
                "- Que competências sustentam os pré-requisitos "
                "de Recuperação Híbrida?\n"
                "- Quais cursos preciso estudar antes de APIs Web?\n"
                "- Como Integração Contínua se conecta a Testes Automatizados?"
            )
        return
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            answer = message.get("answer")
            if answer is not None and all(
                hasattr(answer, field)
                for field in ("answer", "citations", "retrieval", "caveats")
            ):
                render_answer(answer)
            else:
                st.markdown(str(message["content"]))


def render_answer(answer: GroundedAnswer) -> None:
    st.markdown(answer.answer)
    if answer.caveats:
        st.warning(SCOPE_CAVEAT)

    metric_col, evidence_col, graph_col = st.columns(3)
    metric_col.metric("Confiança da recuperação", f"{answer.retrieval.confidence:.0%}")
    evidence_col.metric("Trechos usados", len(answer.retrieval.hits))
    graph_col.metric("Relações no contexto", len(answer.retrieval.edges))

    sources_tab, graph_tab, trace_tab = st.tabs(
        ["📚 Fontes", "🕸️ Subgrafo", "🔎 Rastreabilidade"]
    )
    with sources_tab:
        render_sources(answer)
    with graph_tab:
        render_graph(answer)
    with trace_tab:
        render_trace(answer)


def render_sources(answer: GroundedAnswer) -> None:
    if not answer.retrieval.hits:
        st.info("Nenhuma fonte foi recuperada.")
        return
    for index, hit in enumerate(answer.retrieval.hits, start=1):
        source_id = f"F{index}"
        cited = source_id in answer.citations
        label = "citada" if cited else "contexto adicional"
        with st.expander(f"[{source_id}] {hit.title} · {label}", expanded=index == 1):
            st.caption(
                f"Arquivo: `{hit.filename}` · score RRF: {hit.score:.3f} · "
                f"canais: {', '.join(sorted(hit.channels))}"
            )
            st.write(hit.text)


def render_graph(answer: GroundedAnswer) -> None:
    if not answer.retrieval.nodes:
        st.info("Nenhuma entidade foi encontrada para formar o subgrafo.")
        return
    edges = [
        edge
        for index, edge in enumerate(answer.retrieval.edges, 1)
        if f"G{index}" in getattr(answer, "supporting_edges", [])
    ]
    if not edges:
        st.info("A resposta não declarou relações do grafo como apoio.")
        return
    identifiers = {key for edge in edges for key in (edge.source, edge.target)}
    nodes = [node for node in answer.retrieval.nodes if node.id in identifiers]
    figure = create_graph_figure(nodes, edges)
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )
    labels = sorted({node.label for node in nodes})
    legend = " · ".join(
        f"{label} ({LABEL_COLORS.get(label, '#475569')})" for label in labels
    )
    st.caption(f"Legenda de cores: {legend}")


def render_trace(answer: GroundedAnswer) -> None:
    for warning in getattr(answer.retrieval, "warnings", []):
        st.info(warning)
    st.markdown("**Relações declaradas como apoio da resposta**")
    st.dataframe(
        [
            {
                "id": f"G{index}",
                "origem": edge.source,
                "relação": edge.relationship_type,
                "destino": edge.target,
                "trecho": edge.chunk_id,
            }
            for index, edge in enumerate(answer.retrieval.edges, 1)
            if f"G{index}" in getattr(answer, "supporting_edges", [])
        ],
        hide_index=True,
    )
    st.markdown("**Nós consultados**")
    st.dataframe(
        [
            {"id": node.id, "rótulo": node.label, "nome": node.name}
            for node in answer.retrieval.nodes
        ],
        width="stretch",
        hide_index=True,
    )
    retrieval = answer.retrieval
    if retrieval.paths:
        st.markdown("**Caminho mais curto por pré-requisitos**")
        for path in retrieval.paths:
            st.write(" → ".join(path))
    if retrieval.communities:
        st.markdown("**Comunidades de cursos (Louvain)**")
        for group in retrieval.communities:
            st.write(", ".join(group))
    st.markdown("**Consultas executadas**")
    for index, query in enumerate(answer.retrieval.queries, start=1):
        with st.expander(f"Cypher {index}"):
            st.code(query.strip(), language="cypher")
    st.caption(
        "A confiança é uma heurística de recuperação, não uma probabilidade "
        "estatística de veracidade."
    )


if __name__ == "__main__":
    main()
