"""Enquadra a atividade descrita em linguagem coloquial nas atividades do grafo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from licenciagraph.ingestion.canonicalization import normalize_text
from licenciagraph.ingestion.llm import OpenAIClient
from licenciagraph.retrieval.models import GraphNode, Resolution

MAX_TERMS = 6
MAX_CANDIDATES = 24
MAX_CHOSEN = 6
LEXICAL_SCORE = 0.55
HISTORY_TURNS = 3

INTERPRET_SCHEMA = {
    "type": "object",
    "required": ["pergunta_autonoma", "fora_do_escopo", "tem_atividade", "termos"],
    "properties": {
        "pergunta_autonoma": {"type": "string"},
        "fora_do_escopo": {"type": "boolean"},
        "tem_atividade": {"type": "boolean"},
        "termos": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

INTERPRET_SYSTEM = """Você ajuda empreendedores da Paraíba a achar, no Manual de
Licenciamento Ambiental e na NA-101 da SUDEMA, a atividade que eles querem licenciar.
Eles falam de forma coloquial; as normas usam nomes formais.
1. pergunta_autonoma: reescreva a pergunta atual de modo que seja entendida sem o
   histórico (resolva "e quanto custa?", "isso", "esse caso" usando as mensagens
   anteriores, inclusive números e dados já informados). Sem histórico, repita a
   pergunta. Não responda nada.
2. fora_do_escopo: true se a pergunta trata do licenciamento de outro estado, da
   esfera federal ou municipal, ou de outro órgão que não a SUDEMA (por exemplo
   órgãos ambientais de outros estados e o IBAMA).
3. tem_atividade: true se a pergunta trata de uma atividade, empreendimento ou
   negócio concreto a licenciar; false para perguntas gerais (o que é uma licença,
   quais órgãos existem, o que é UFR-PB).
4. termos: de 3 a 6 nomes formais, curtos, de como a norma poderia listar essa
   atividade, do mais específico ao mais amplo, incluindo a categoria em que ela
   caberia e a atividade ambientalmente relevante por trás do negócio (o que ele
   cria, queima, lava, armazena, descarta ou gera de resíduo). Troque a fala
   popular pelo termo técnico. Quando a pessoa fala em vender ou oferecer um
   animal, produto ou insumo, inclua tanto a atividade de produzi-lo ou criá-lo
   quanto a de apenas revendê-lo. Exemplos: "granja de frango" -> avicultura industrial,
   criação de aves, postura e corte; "lava-jato" -> lavagem de veículos, serviço de
   lavagem e lubrificação; "padaria" -> panificação, fabricação de produtos de
   padaria, comércio varejista de alimentos. Se não houver atividade, termos vazio."""

CHOOSE_SYSTEM = """Você enquadra a atividade descrita por um empreendedor entre as
atividades listadas pela SUDEMA. Recebe a pergunta, a atividade descrita e uma lista
numerada de CANDIDATAS (nome, regime e grupo, tirados do Manual e da NA-101).
tipo:
- exata: há candidata que nomeia o mesmo tipo de negócio, produto ou serviço. Família
  parecida que produz ou presta outra coisa não é exata (é proxima). Escolha todas as
  candidatas que cobrem o caso, inclusive as variações por porte ou quantidade
  (por exemplo, faixas de número de animais ou de área útil), pois o porte decide
  o regime. Se a fala é ambígua (por exemplo, "vender" um produto pode ser produzi-lo
  ou apenas revendê-lo), escolha as candidatas de cada leitura plausível, use exata
  quando alguma delas for do mesmo tipo de atividade e liste em faltam o dado que
  decide qual leitura vale.
- proxima: nenhuma candidata é a atividade, mas algumas têm natureza, impacto ou
  função parecidos e podem servir de referência. Escolha só as mais parecidas.
- nenhuma: nenhuma candidata guarda relação razoável com a atividade.
Use apenas números da lista. Não invente candidatas. motivo: uma frase, em
português, dizendo por que as escolhidas se aplicam e, em proxima, por que a
atividade não consta nominalmente. faltam: de 0 a 3 informações curtas que o
empreendedor precisa dar para fechar o enquadramento (por exemplo, quantidade,
área útil, se gera determinado resíduo); vazio se nada falta ou se nenhuma."""


def _choose_schema(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["tipo", "escolhidas", "motivo", "faltam"],
        "properties": {
            "tipo": {"type": "string", "enum": ["exata", "proxima", "nenhuma"]},
            "escolhidas": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": max(count, 1)},
            },
            "motivo": {"type": "string"},
            "faltam": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class Candidate:
    node: GraphNode
    code: str
    group: str
    regimes: tuple[str, ...]
    score: float

    def line(self, number: int) -> str:
        regimes = ", ".join(self.regimes) or "sem regime listado"
        return (
            f"[{number}] {self.node.name} (código {self.code}; {regimes}; {self.group})"
        )


class ActivityResolver:
    def __init__(
        self,
        llm: OpenAIClient,
        repository: Any,
        *,
        model: str,
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        self.llm = llm
        self.repository = repository
        self.model = model
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension

    def resolve(
        self, question: str, history: list[tuple[str, str]] | None = None
    ) -> Resolution:
        turns = [
            {"pergunta": q[:400], "resposta": a[:400]}
            for q, a in (history or [])[-HISTORY_TURNS:]
        ]
        raw = self.llm.chat_json(
            model=self.model,
            system=INTERPRET_SYSTEM,
            prompt=json.dumps(
                {"historico": turns, "pergunta": question}, ensure_ascii=False
            ),
            schema=INTERPRET_SCHEMA,
            strict=True,
        )
        standalone = str(raw.get("pergunta_autonoma") or question).strip() or question
        terms = tuple(
            dict.fromkeys(t.strip() for t in raw.get("termos", []) if str(t).strip())
        )[:MAX_TERMS]
        if raw.get("fora_do_escopo"):
            return Resolution("fora_do_escopo", standalone=standalone)
        if not raw.get("tem_atividade") or not terms:
            return Resolution("nenhuma", standalone=standalone, terms=terms)
        candidates = self._candidates(standalone, terms)
        if not candidates:
            return Resolution("nenhuma", standalone=standalone, terms=terms)
        return self._choose(standalone, terms, candidates)

    def _candidates(self, question: str, terms: tuple[str, ...]) -> list[Candidate]:
        texts = [question, *terms]
        vectors = self.llm.embed(
            model=self.embedding_model,
            texts=texts,
            dimensions=self.embedding_dimension,
        )
        scores: dict[str, float] = {}
        for index, vector in enumerate(vectors):
            weight = 0.9 if index == 0 else 1.0
            for node_id, score in self.repository.activity_scores(vector, limit=15):
                scores[node_id] = max(scores.get(node_id, 0.0), score * weight)
        for node_id in self.repository.activity_ids_by_stem(stems(terms)):
            scores[node_id] = max(scores.get(node_id, 0.0), LEXICAL_SCORE)
        ranked = sorted(scores.items(), key=lambda item: -item[1])[:MAX_CANDIDATES]
        details = self.repository.activity_details([node_id for node_id, _ in ranked])
        return [
            Candidate(
                GraphNode(node_id, "Atividade", details[node_id]["name"]),
                details[node_id]["code"],
                details[node_id]["group"],
                tuple(details[node_id]["regimes"]),
                score,
            )
            for node_id, score in ranked
            if node_id in details
        ]

    def _choose(
        self, standalone: str, terms: tuple[str, ...], candidates: list[Candidate]
    ) -> Resolution:
        listing = "\n".join(c.line(i) for i, c in enumerate(candidates, 1))
        raw = self.llm.chat_json(
            model=self.model,
            system=CHOOSE_SYSTEM,
            prompt=(
                f"PERGUNTA: {standalone}\nTERMOS FORMAIS: {', '.join(terms)}\n\n"
                f"CANDIDATAS:\n{listing}"
            ),
            schema=_choose_schema(len(candidates)),
            strict=True,
        )
        picked = [
            candidates[i - 1].node
            for i in dict.fromkeys(raw.get("escolhidas", []))
            if 1 <= i <= len(candidates)
        ][:MAX_CHOSEN]
        match = raw.get("tipo", "nenhuma")
        if match != "nenhuma" and not picked:
            match = "nenhuma"
        if match == "nenhuma":
            picked = []
        missing = tuple(m.strip() for m in raw.get("faltam", []) if str(m).strip())[:3]
        return Resolution(
            match,
            tuple(picked),
            str(raw.get("motivo", "")).strip(),
            missing if picked else (),
            terms,
            standalone,
        )


GENERIC_WORDS = frozenset(
    "atividade atividades servico servicos comercio fabricacao empreendimento "
    "empreendimentos producao industrial industria prestacao varejista "
    "equipamentos estabelecimento".split()
)


def stems(terms: tuple[str, ...]) -> list[str]:
    """Radicais de busca lexical: a primeira palavra específica de cada termo."""
    result = []
    for term in terms:
        for word in normalize_text(term).split():
            if len(word) >= 6 and word not in GENERIC_WORDS:
                result.append(word[: max(5, len(word) - 3)])
                break
    return list(dict.fromkeys(result))
