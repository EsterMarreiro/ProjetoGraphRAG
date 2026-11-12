"""Aplicacao Streamlit do LicenciaGraph."""

from __future__ import annotations

import streamlit as st

from licenciagraph.guide import (
    ActivityOption,
    ActivityProfile,
    Band,
    LicensingGuide,
    band_for_value,
    checklist_markdown,
)
from licenciagraph.retrieval.generation import SCOPE_CAVEAT
from licenciagraph.retrieval.models import GroundedAnswer
from licenciagraph.ui.filters import ProcessFilters
from licenciagraph.ui.graph_view import LABEL_COLORS, create_graph_figure
from licenciagraph.ui.runtime import UIRuntime, build_runtime

MODALITIES = (
    "",
    "Dispensa",
    "LAC",
    "LAU",
    "AA",
    "LP",
    "LI",
    "LO",
    "RLI",
    "RLO",
    "LAI",
    "LAO",
    "LOC",
)
PORTES = ("", "Micro", "Pequeno", "Médio", "Grande", "Extraordinário")
EXAMPLES = (
    "Quais documentos preciso para a Licença de Instalação de uma avicultura "
    "industrial?",
    "Qual a validade e o custo em UFR-PB da LAU para avicultura industrial de "
    "porte médio?",
    "Qual a sequência entre Licença Prévia, de Instalação e de Operação?",
    "Quais atividades de comércio e serviços de saúde são dispensadas de "
    "licenciamento?",
    "Vou abrir um estúdio de tatuagem. Preciso de licença ambiental?",
    "Quero criar e vender porcos. Qual licença preciso?",
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
        page_title="LicenciaGraph",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🌿 LicenciaGraph")
    st.caption(
        "Guia de licenciamento ambiental da SUDEMA (Paraíba), com evidências e grafos"
    )

    try:
        runtime = get_runtime()
    except Exception as exc:
        st.error(f"Não foi possível conectar ao FalkorDB: {exc}")
        st.info("Inicie a infraestrutura com `docker compose up -d falkordb`.")
        st.stop()

    filters, services_ready = render_sidebar(runtime)
    chat_tab, guide_tab = st.tabs(["💬 Perguntar", "🧭 Assistente de enquadramento"])
    with chat_tab:
        render_chat(runtime, filters, services_ready)
    with guide_tab:
        render_guide(runtime.guide, runtime.service.retriever.resolver)


def render_chat(
    runtime: UIRuntime, filters: ProcessFilters, services_ready: bool
) -> None:
    initialize_history()
    render_history()
    prompt = st.chat_input(
        "Pergunte sobre licenças, documentos, porte, custos ou prazos...",
        disabled=not services_ready,
    )
    if not prompt:
        return
    history = conversation_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    augmented_question = filters.augment_question(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando evidências e percorrendo o grafo..."):
                answer = runtime.service.answer(augmented_question, history)
            render_answer(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer.answer, "answer": answer}
            )
        except Exception as exc:
            message = f"Não foi possível responder: {exc}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})


def render_sidebar(runtime: UIRuntime) -> tuple[ProcessFilters, bool]:
    with st.sidebar:
        st.header("Contexto da pergunta")
        activity = st.text_input("Atividade (opcional)")
        modality = st.selectbox("Modalidade", MODALITIES)
        porte = st.selectbox("Porte", PORTES)

        st.divider()
        st.subheader("Estado dos serviços")
        database_ok = _database_status(runtime)
        llm_ok = _llm_status(runtime)

        st.divider()
        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption(
            "Base: Manual de Licenciamento Ambiental (5ª ed.) e NA-101 (5ª ed., 2026) "
            "da SUDEMA. Orientação; não substitui a análise do órgão ambiental."
        )

    return (
        ProcessFilters(activity, modality, porte),
        database_ok and llm_ok,
    )


def render_guide(guide: LicensingGuide, resolver=None) -> None:
    st.markdown(
        "Informe a atividade e o porte para ver a modalidade, o custo de análise "
        "(em UFR-PB), a validade e a lista de documentos. A busca por nome consulta "
        "o grafo diretamente, sem modelo de linguagem."
    )
    query = st.text_input("Atividade (nome ou código)", key="guide_query")
    options = guide.search_activities(query) if query.strip() else []
    if query.strip() and not options:
        options = suggest_by_description(guide, resolver, query)
        if not options:
            return
    if not options:
        return
    chosen = st.selectbox(
        "Atividades encontradas",
        options,
        format_func=lambda option: f"{option.code} · {option.name}",
        key="guide_option",
    )
    profile = guide.profile(chosen.id)
    if profile is None:
        st.warning("Atividade não encontrada no grafo.")
        return
    st.subheader(profile.name)
    st.caption(f"Código {profile.code} · {profile.group} · {profile.subgroup}")
    render_regimes(profile)
    band = choose_band(profile)
    render_band(guide, profile, band)
    render_checklists(guide, profile)
    st.caption(SCOPE_CAVEAT)


def suggest_by_description(guide, resolver, text: str) -> list:
    """Sem nome correspondente, oferece interpretar a descrição com a IA."""
    st.info(
        "Nenhuma atividade tem esse termo no nome. Você pode descrever o negócio com "
        "suas palavras e deixar a IA indicar as atividades do Manual mais próximas."
    )
    if resolver is None:
        return []
    cached = st.session_state.get("guide_ai")
    if st.button("Interpretar com IA (usa a API)", key="guide_ai_button"):
        with st.spinner("Procurando as atividades mais próximas..."):
            resolution = resolver.resolve(text)
        profiles = [guide.profile(node.id) for node in resolution.activities]
        options = [ActivityOption(p.id, p.code, p.name, p.group) for p in profiles if p]
        cached = (text, resolution.match, options, resolution.missing)
        st.session_state["guide_ai"] = cached
    if not cached or cached[0] != text:
        return []
    _, match, options, missing = cached
    if not options:
        st.warning("Nenhuma atividade parecida foi encontrada nas normas.")
        return []
    if match == "proxima":
        st.warning(
            "A atividade não consta nominalmente; estas são as mais próximas. "
            "Confirme o enquadramento com a SUDEMA."
        )
    if missing:
        st.caption("Para fechar o enquadramento: " + "; ".join(missing))
    return options


def render_regimes(profile: ActivityProfile) -> None:
    if not profile.regimes:
        st.info("Nenhuma modalidade de licenciamento consta para esta atividade.")
        return
    st.markdown("**Modalidades possíveis**")
    st.dataframe(
        [
            {
                "sigla": regime.modality,
                "modalidade": regime.name,
                "validade": regime.validity.lower(),
                "letra de custo": regime.letter,
            }
            for regime in profile.regimes
        ],
        hide_index=True,
        width="stretch",
    )


def choose_band(profile: ActivityProfile) -> Band | None:
    bands = [band for band in profile.bands if band.criterion or band.letter]
    if not bands:
        return None
    st.markdown("**Porte**")
    unit = profile.porte_unit or "critério do porte"
    has_ranges = any(band.interval for band in bands)
    value = (
        st.number_input(f"Informe o valor ({unit})", min_value=0.0, value=0.0)
        if has_ranges
        else 0.0
    )
    matched = band_for_value(bands, value) if has_ranges and value > 0 else None
    labels = [band.label for band in bands]
    default = labels.index(matched.label) if matched else 0
    label = st.radio("Porte aplicado", labels, index=default, horizontal=True)
    return next(band for band in bands if band.label == label)


def render_band(
    guide: LicensingGuide, profile: ActivityProfile, band: Band | None
) -> None:
    if band is None:
        return
    cols = st.columns(3)
    cols[0].metric("Porte", band.label)
    cols[1].metric("Letra de custo", band.letter or "n/d")
    cols[2].markdown(
        "**Potencial poluidor**\n\n"
        f"ar {band.air or '-'} · água {band.water or '-'} · solo {band.soil or '-'}"
    )
    if band.note:
        st.info(f"Anotação da NA-101 para este porte: {band.note}")
    modalities = sorted(
        {regime.modality for regime in profile.regimes}
        | {checklist.modality for checklist in profile.checklists}
    )
    costs = guide.costs(band.letter, modalities)
    if costs:
        st.markdown("**Custo de análise (UFR-PB) por modalidade**")
        st.dataframe(
            [
                {"modalidade": modality, "UFR-PB": costs[modality]}
                for modality in modalities
                if modality in costs
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Os valores estão em UFR-PB; a conversão para reais depende do valor "
            "vigente da UFR-PB."
        )


def render_checklists(guide: LicensingGuide, profile: ActivityProfile) -> None:
    if not profile.checklists:
        return
    st.markdown("**Documentos exigidos por checklist**")
    for checklist in profile.checklists:
        with st.expander(f"{checklist.title} · p. {checklist.pages}"):
            items = guide.requirements(checklist.id)
            if not items:
                st.info("Itens do checklist indisponíveis.")
                continue
            for item in items:
                extra = []
                if item.agency:
                    extra.append(f"órgão: {item.agency}")
                if item.condition:
                    extra.append(item.condition)
                suffix = f" _({'; '.join(extra)})_" if extra else ""
                st.markdown(f"{item.number}. {item.text}{suffix}")
            st.download_button(
                "Baixar checklist (Markdown)",
                checklist_markdown(profile, checklist, items),
                file_name=f"{checklist.code}.md",
                key=f"download-{checklist.id}",
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


def conversation_history(messages: list[dict]) -> list[tuple[str, str]]:
    """Pares (pergunta, resposta) já trocados, para entender perguntas de seguimento."""
    pairs = []
    for user, assistant in zip(messages[::2], messages[1::2], strict=False):
        if user["role"] == "user" and assistant["role"] == "assistant":
            pairs.append((str(user["content"]), str(assistant["content"])))
    return pairs


def initialize_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_history() -> None:
    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("**Experimente perguntar:**")
            st.markdown("\n".join(f"- {example}" for example in EXAMPLES))
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

    render_resolution(answer)
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


def render_resolution(answer: GroundedAnswer) -> None:
    resolution = answer.retrieval.resolution
    if resolution is None or not resolution.activities:
        return
    kind = "mesma atividade" if resolution.match == "exata" else "atividade próxima"
    names = "; ".join(node.name for node in resolution.activities)
    st.info(f"**Enquadramento interpretado ({kind}):** {names}", icon="🧭")


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
    if retrieval.resolution and retrieval.resolution.terms:
        st.markdown("**Termos formais derivados da pergunta**")
        st.write(", ".join(retrieval.resolution.terms))
    if retrieval.paths:
        st.markdown("**Sequência de licenças (caminho mais curto)**")
        for path in retrieval.paths:
            st.write(" → ".join(path))
    if retrieval.communities:
        st.markdown("**Atividades com checklists em comum (Louvain)**")
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
