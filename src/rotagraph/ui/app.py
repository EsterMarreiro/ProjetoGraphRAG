"""Aplicacao Streamlit do RotaGraph Brasil."""

from __future__ import annotations

import streamlit as st

from rotagraph.retrieval.models import GroundedAnswer
from rotagraph.ui.filters import TourismFilters
from rotagraph.ui.graph_view import LABEL_COLORS, create_graph_figure
from rotagraph.ui.runtime import UIRuntime, build_runtime

REGIONS = ("Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul")
ACTIVITIES = (
    "",
    "Caminhada",
    "Ecoturismo",
    "Fotografia",
    "Gastronomia",
    "Navegação",
    "Patrimônio histórico",
)
PROFILES = (
    "",
    "Aventura moderada",
    "Ecoturista",
    "Estudantes",
    "Famílias",
    "Fotógrafos",
    "História e cultura",
)
PERIODS = (
    "",
    "Período seco",
    "Período chuvoso",
    "Temperaturas amenas",
    "Lagoas abastecidas",
    "Águas baixas",
)


@st.cache_resource(
    scope="session",
    show_spinner="Conectando aos serviços...",
    on_release=lambda runtime: runtime.close(),
)
def get_runtime() -> UIRuntime:
    return build_runtime()


def main() -> None:
    st.set_page_config(
        page_title="RotaGraph Brasil",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🧭 RotaGraph Brasil")
    st.caption(
        "Assistente de turismo fundamentado em busca híbrida e grafo de conhecimento"
    )

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
        "Pergunte sobre destinos, atrações, atividades ou rotas...",
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


def render_sidebar(runtime: UIRuntime) -> tuple[TourismFilters, bool]:
    with st.sidebar:
        st.header("Filtros da viagem")
        regions = tuple(st.multiselect("Regiões", REGIONS))
        state = st.text_input("UF ou estado", placeholder="Ex.: BA")
        activity = st.selectbox("Atividade", ACTIVITIES)
        profile = st.selectbox("Perfil", PROFILES)
        period = st.selectbox("Período", PERIODS)

        st.divider()
        st.subheader("Estado dos serviços")
        database_ok = _database_status(runtime)
        ollama_ok = _ollama_status(runtime)

        st.divider()
        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption(
            "As recomendações usam somente o corpus local. Confirme condições, "
            "acessos e regras antes da viagem."
        )

    return (
        TourismFilters(regions, state, activity, profile, period),
        database_ok and ollama_ok,
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


def _ollama_status(runtime: UIRuntime) -> bool:
    try:
        models = runtime.ollama.list_models()
    except Exception as exc:
        st.error(f"Ollama indisponível: {exc}", icon="🚨")
        return False
    missing = [
        model
        for model in (runtime.settings.chat_model, runtime.settings.embedding_model)
        if model not in models
    ]
    if missing:
        st.warning(f"Modelos ausentes no Ollama: {', '.join(missing)}", icon="⚠️")
        return False
    st.success("Ollama e modelos disponíveis", icon="✅")
    return True


def initialize_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_history() -> None:
    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("**Experimente perguntar:**")
            st.markdown(
                "- Qual rota combina patrimônio histórico e gastronomia?\n"
                "- Onde fazer ecoturismo com atividades adequadas para famílias?\n"
                "- Que cuidados sazonais são citados para a Chapada dos Veadeiros?"
            )
        return
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            answer = message.get("answer")
            if isinstance(answer, GroundedAnswer):
                render_answer(answer)
            else:
                st.markdown(str(message["content"]))


def render_answer(answer: GroundedAnswer) -> None:
    st.markdown(answer.answer)
    if answer.caveats:
        st.warning("\n".join(f"• {item}" for item in answer.caveats))

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
    figure = create_graph_figure(answer.retrieval.nodes, answer.retrieval.edges)
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )
    labels = sorted({node.label for node in answer.retrieval.nodes})
    legend = " · ".join(
        f"{label} ({LABEL_COLORS.get(label, '#475569')})" for label in labels
    )
    st.caption(f"Legenda de cores: {legend}")


def render_trace(answer: GroundedAnswer) -> None:
    st.markdown("**Nós consultados**")
    st.dataframe(
        [
            {"id": node.id, "rótulo": node.label, "nome": node.name}
            for node in answer.retrieval.nodes
        ],
        width="stretch",
        hide_index=True,
    )
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
