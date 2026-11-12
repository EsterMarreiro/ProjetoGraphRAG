"""Aplicacao Streamlit do LicenciaGraph."""

from __future__ import annotations

import html
import time

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
from licenciagraph.ui.theme import active_css, apply_theme, render_brand

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
CHAT_PAGE = "💬 Conversa"
GUIDE_PAGE = "🧭 Assistente de enquadramento"
EXAMPLES = (
    "Vou abrir uma padaria. Preciso de licença ambiental?",
    "Quero criar e vender porcos. Qual licença preciso?",
    "Vou abrir um estúdio de tatuagem. Preciso de licença?",
    "Qual a diferença entre Licença Prévia, de Instalação e de Operação?",
    "Quais documentos preciso para a Licença de Instalação de uma avicultura "
    "industrial?",
    "Quais atividades de comércio e serviços de saúde são dispensadas de "
    "licenciamento?",
)
HERO_TITLE = "Como posso ajudar com o seu licenciamento ambiental?"
HERO_TEXT = (
    "Tire dúvidas sobre licenças, documentos, porte, custos e prazos da SUDEMA em "
    "linguagem simples, sem precisar procurar em centenas de páginas do Manual e "
    "da NA-101. Conte o que você quer licenciar; cada resposta mostra a fonte e a "
    "página."
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
        page_title="LicenciaGraph · SUDEMA",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    init_state()

    try:
        runtime = get_runtime()
    except Exception as exc:
        st.error(f"Não foi possível conectar ao FalkorDB: {exc}")
        st.info("Inicie a infraestrutura com `docker compose up -d falkordb`.")
        st.stop()

    status = service_status(runtime)
    render_sidebar(runtime, status)
    if st.session_state.page == GUIDE_PAGE:
        render_guide(runtime.guide, runtime.service.retriever.resolver)
    else:
        render_chat(runtime, current_filters(), status["ready"])


def init_state() -> None:
    state = st.session_state
    state.setdefault("conversations", {})
    state.setdefault("counter", 0)
    state.setdefault("page", CHAT_PAGE)
    state.setdefault("ctx", {})
    if state.get("current") not in state.conversations:
        start_new_conversation()


def start_new_conversation() -> None:
    """Reaproveita a conversa atual se estiver vazia; senão abre uma nova."""
    state = st.session_state
    current = state.conversations.get(state.get("current"))
    state.page = CHAT_PAGE
    if current is not None and not current["messages"]:
        return
    state.counter += 1
    state.current = f"c{state.counter}"
    state.conversations[state.current] = {"title": "Nova conversa", "messages": []}


def select_conversation(conversation_id: str) -> None:
    st.session_state.current = conversation_id
    st.session_state.page = CHAT_PAGE


def delete_conversation(conversation_id: str) -> None:
    state = st.session_state
    state.conversations.pop(conversation_id, None)
    if state.current == conversation_id:
        state.pop("current", None)


def go_to(page: str) -> None:
    st.session_state.page = page


def current_messages() -> list[dict]:
    state = st.session_state
    return state.conversations[state.current]["messages"]


def current_filters() -> ProcessFilters:
    ctx = st.session_state.ctx
    return ProcessFilters(
        ctx.get("activity", ""), ctx.get("modality", ""), ctx.get("porte", "")
    )


def service_status(runtime: UIRuntime, ttl: float = 60.0) -> dict:
    """Estado do banco e da API, consultado no máximo uma vez por ttl segundos."""
    cached = st.session_state.get("services")
    if cached and time.monotonic() - cached["at"] < ttl:
        return cached
    db_ok, db_message = _database_status(runtime)
    llm_ok, llm_message = _llm_status(runtime)
    status = {
        "at": time.monotonic(),
        "ready": db_ok and llm_ok,
        "messages": [(db_ok, db_message), (llm_ok, llm_message)],
    }
    st.session_state.services = status
    return status


def render_chat(
    runtime: UIRuntime, filters: ProcessFilters, services_ready: bool
) -> None:
    messages = current_messages()
    if messages:
        render_history(messages)
    else:
        render_hero()
    typed = st.chat_input(
        "Pergunte sobre licenças, documentos, porte, custos ou prazos...",
        disabled=not services_ready,
    )
    prompt = typed or st.session_state.pop("pending_prompt", None)
    if not prompt or not services_ready:
        return
    conversation = st.session_state.conversations[st.session_state.current]
    history = conversation_history(messages)
    if not messages:
        conversation["title"] = prompt.strip()[:40]
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    augmented_question = filters.augment_question(prompt)
    with st.chat_message("assistant", avatar="🌿"):
        try:
            with st.spinner("Interpretando a sua atividade e consultando as normas..."):
                answer = runtime.service.answer(augmented_question, history)
            render_answer(answer)
            messages.append(
                {"role": "assistant", "content": answer.answer, "answer": answer}
            )
        except Exception as exc:
            message = f"Não foi possível responder: {exc}"
            st.error(message)
            messages.append({"role": "assistant", "content": message, "error": True})
    st.rerun()


def render_hero() -> None:
    st.markdown(
        f'<div class="hero"><h1>{HERO_TITLE}</h1><p>{HERO_TEXT}</p></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(2)
    for index, example in enumerate(EXAMPLES):
        columns[index % 2].button(
            example,
            key=f"example-{index}",
            use_container_width=True,
            on_click=lambda text=example: st.session_state.update(pending_prompt=text),
        )


def render_sidebar(runtime: UIRuntime, status: dict) -> None:
    state = st.session_state
    with st.sidebar:
        render_brand()
        st.button(
            "Nova conversa",
            key="new_chat",
            use_container_width=True,
            on_click=start_new_conversation,
        )
        st.button(
            "Conversa",
            key="nav_chat",
            use_container_width=True,
            on_click=go_to,
            args=(CHAT_PAGE,),
        )
        st.button(
            "Assistente de enquadramento",
            key="nav_guide",
            use_container_width=True,
            on_click=go_to,
            args=(GUIDE_PAGE,),
        )

        history = [
            (cid, conv)
            for cid, conv in reversed(list(state.conversations.items()))
            if conv["messages"]
        ]
        if history:
            st.markdown(
                '<div class="side-label">Conversas</div>', unsafe_allow_html=True
            )
        for cid, conv in history:
            title_col, delete_col = st.columns([6, 1])
            title_col.button(
                conv["title"],
                key=f"conv_{cid}",
                use_container_width=True,
                on_click=select_conversation,
                args=(cid,),
            )
            delete_col.button(
                " ",
                key=f"del_{cid}",
                help="Apagar conversa",
                on_click=delete_conversation,
                args=(cid,),
            )

        with st.container(key="sidebar_footer"):
            st.divider()
            ok = status["ready"]
            st.markdown(
                f'<div class="status"><i class="{"" if ok else "off"}"></i>'
                f"{'Serviços ativos' if ok else 'Serviço indisponível'}</div>",
                unsafe_allow_html=True,
            )
            if st.button(
                "Configurações", key="open_settings", use_container_width=True
            ):
                settings_dialog(status)

    active = ["nav_guide" if state.page == GUIDE_PAGE else "nav_chat"]
    if state.page != GUIDE_PAGE and state.conversations[state.current]["messages"]:
        active.append(f"conv_{state.current}")
    active_css(active)


@st.dialog("Configurações", width="large")
def settings_dialog(status: dict) -> None:
    context_tab, services_tab, about_tab = st.tabs(
        ["Contexto da pergunta", "Serviços", "Sobre"]
    )
    ctx = st.session_state.ctx
    with context_tab:
        st.caption(
            "Opcional. Estes dados são acrescentados às suas perguntas para ajudar "
            "no enquadramento."
        )
        activity = st.text_input("Atividade", value=ctx.get("activity", ""))
        modality = st.selectbox(
            "Modalidade",
            MODALITIES,
            index=MODALITIES.index(ctx.get("modality", "")),
            format_func=lambda value: value or "Qualquer",
        )
        porte = st.selectbox(
            "Porte",
            PORTES,
            index=PORTES.index(ctx.get("porte", "")),
            format_func=lambda value: value or "Qualquer",
        )
        st.session_state.ctx = {
            "activity": activity,
            "modality": modality,
            "porte": porte,
        }
    with services_tab:
        for ok, message in status["messages"]:
            (st.success if ok else st.error)(message)
        if st.button("Verificar de novo"):
            st.session_state.pop("services", None)
            st.rerun()
    with about_tab:
        st.markdown(
            "**LicenciaGraph** ajuda o empreendedor a entender o licenciamento "
            "ambiental da SUDEMA na Paraíba: qual licença a atividade exige, quais "
            "documentos levar, o porte, o custo de análise em UFR-PB e a validade."
        )
        st.markdown(
            "Base: Manual de Licenciamento Ambiental (5ª ed.) e NA-101 (5ª ed., "
            "2026) da SUDEMA. É uma ferramenta de apoio: não é um canal oficial e "
            "não substitui a análise do órgão ambiental."
        )
        st.caption(
            "As conversas ficam só nesta sessão do navegador; nada é gravado em disco."
        )
        if st.button("Apagar todas as conversas"):
            st.session_state.conversations = {}
            st.session_state.pop("current", None)
            st.rerun()


def render_guide(guide: LicensingGuide, resolver=None) -> None:
    st.markdown("## 🧭 Assistente de enquadramento")
    st.markdown(
        "Em três passos: diga a atividade, informe o porte e veja a modalidade, o "
        "custo de análise (em UFR-PB), a validade e a lista de documentos. A busca "
        "por nome consulta o grafo diretamente, sem modelo de linguagem."
    )
    st.markdown("#### 1. Qual é a sua atividade?")
    query = st.text_input(
        "Nome, código ou descrição com as suas palavras",
        key="guide_query",
        placeholder="Ex.: avicultura, 49.65.430, criação de porcos",
    )
    options = guide.search_activities(query) if query.strip() else []
    if query.strip():
        options = (
            suggest_by_description(guide, resolver, query, found=bool(options))
            or options
        )
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
    st.markdown("#### 2. Porte e modalidade")
    st.markdown(f"**{profile.name}**")
    st.caption(f"Código {profile.code} · {profile.group} · {profile.subgroup}")
    render_regimes(profile)
    band = choose_band(profile)
    st.markdown("#### 3. Custo e documentos")
    render_band(guide, profile, band)
    render_checklists(guide, profile)
    st.caption(SCOPE_CAVEAT)


def suggest_by_description(guide, resolver, text: str, found: bool) -> list:
    """Interpreta com a IA a descrição do negócio e sugere atividades do Manual."""
    if not found:
        st.info(
            "Nenhuma atividade tem esse termo no nome. Descreva o negócio com suas "
            "palavras e deixe a IA indicar as atividades do Manual mais próximas."
        )
    if resolver is None:
        return []
    cached = st.session_state.get("guide_ai")
    label = (
        "Interpretar com IA (usa a API)"
        if not found
        else "Não é isso? Interpretar com IA"
    )
    if st.button(label, key="guide_ai_button"):
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


def _database_status(runtime: UIRuntime) -> tuple[bool, str]:
    try:
        runtime.connection.healthcheck()
    except Exception as exc:
        return False, f"FalkorDB indisponível: {exc}"
    return True, f"FalkorDB conectado · {runtime.settings.falkordb_graph}"


def _llm_status(runtime: UIRuntime) -> tuple[bool, str]:
    try:
        models = runtime.llm.list_models()
    except Exception as exc:
        return False, f"OpenAI indisponível: {exc}"
    missing = [
        model
        for model in (runtime.settings.chat_model, runtime.settings.embedding_model)
        if model not in models
    ]
    if missing:
        return False, f"Modelos indisponíveis na OpenAI: {', '.join(missing)}"
    return True, "OpenAI e modelos disponíveis"


def conversation_history(messages: list[dict]) -> list[tuple[str, str]]:
    """Pares (pergunta, resposta) já trocados, para entender perguntas de seguimento."""
    pairs = []
    for user, assistant in zip(messages[::2], messages[1::2], strict=False):
        if user["role"] == "user" and assistant["role"] == "assistant":
            pairs.append((str(user["content"]), str(assistant["content"])))
    return pairs


def render_history(messages: list[dict]) -> None:
    for message in messages:
        role = message["role"]
        with st.chat_message(role, avatar="🌿" if role == "assistant" else None):
            answer = message.get("answer")
            if message.get("error"):
                st.error(str(message["content"]))
            elif answer is not None and all(
                hasattr(answer, field)
                for field in ("answer", "citations", "retrieval", "caveats")
            ):
                render_answer(answer)
            else:
                st.markdown(str(message["content"]))


def render_answer(answer: GroundedAnswer) -> None:
    st.markdown(answer.answer)
    render_resolution(answer)
    retrieval = answer.retrieval
    if retrieval.hits:
        with st.expander("Fontes, subgrafo e rastreabilidade"):
            metric_col, evidence_col, graph_col = st.columns(3)
            metric_col.metric("Confiança da recuperação", f"{retrieval.confidence:.0%}")
            evidence_col.metric("Trechos usados", len(retrieval.hits))
            graph_col.metric("Relações no contexto", len(retrieval.edges))
            sources_tab, graph_tab, trace_tab = st.tabs(
                ["📚 Fontes", "🕸️ Subgrafo", "🔎 Rastreabilidade"]
            )
            with sources_tab:
                render_sources(answer)
            with graph_tab:
                render_graph(answer)
            with trace_tab:
                render_trace(answer)
    if answer.caveats:
        st.markdown(f'<div class="note">{SCOPE_CAVEAT}</div>', unsafe_allow_html=True)


def render_resolution(answer: GroundedAnswer) -> None:
    resolution = answer.retrieval.resolution
    if resolution is None or not resolution.activities:
        return
    kind = "mesma atividade" if resolution.match == "exata" else "atividade próxima"
    names = "; ".join(html.escape(node.name) for node in resolution.activities)
    st.markdown(
        f'<span class="chip">🧭 Enquadramento interpretado ({kind}): {names}</span>',
        unsafe_allow_html=True,
    )
    if resolution.reason:
        st.caption(f"Como interpretei: {resolution.reason}")


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
