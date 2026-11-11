"""Plano semântico por LLM compilado para Cypher parametrizado e somente leitura."""

from __future__ import annotations

import json
from dataclasses import dataclass

from trilhagraph.ingestion.canonicalization import normalize_text
from trilhagraph.ingestion.models import ALLOWED_LABELS, RELATIONSHIP_ENDPOINTS
from trilhagraph.ingestion.ollama import OllamaClient

QUERY_BY_TYPES = {
    (label, relationship): f"MATCH p=(a:{label})-[:{relationship}*1..2]-(b) "
    "WHERE toLower(a.nome) CONTAINS toLower($name) "
    "UNWIND relationships(p) AS r "
    "RETURN DISTINCT r.trecho_id AS chunk_id LIMIT 20"
    for relationship, endpoints in sorted(RELATIONSHIP_ENDPOINTS.items())
    for label in sorted(set(endpoints))
}
CYPHER_CHOICES = tuple(QUERY_BY_TYPES.values())
PLAN_SCHEMA = {
    "type": "object",
    "required": ["label", "relationship", "name"],
    "properties": {
        "label": {"type": "string", "enum": ["", *sorted(ALLOWED_LABELS)]},
        "relationship": {
            "type": "string",
            "enum": ["", *sorted(RELATIONSHIP_ENDPOINTS)],
        },
        "name": {"type": "string"},
    },
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class CypherPlan:
    cypher: str
    name: str


class CypherPlanner:
    def __init__(self, client: OllamaClient, *, model: str) -> None:
        self.client = client
        self.model = model

    def plan(self, question: str) -> CypherPlan | None:
        prompt = json.dumps({"pergunta": question}, ensure_ascii=False)
        system = """Planeje uma consulta ao catálogo de cursos.
Retorne label, relationship e name para uma entidade explicitamente nomeada na
pergunta. name deve ser copiado literalmente da pergunta. Não invente nomes.
Tipos: Curso=disciplinas; Competencia=habilidades; Area=áreas de conhecimento;
Trilha=percursos temáticos; Perfil=públicos; Nivel=dificuldade.
Relações: Curso ABORDA Area; Curso DESENVOLVE Competencia;
Curso PRE_REQUISITO Curso (o primeiro exige o segundo);
Trilha INCLUI Curso; Curso INDICADO_PARA Perfil; Curso CLASSIFICADO_EM Nivel.
Escolha uma relação que toque o tipo da entidade nomeada e ajude a pergunta.
Exemplo: "Qual curso desenvolve Recuperação vetorial?" -> label Competencia,
relationship DESENVOLVE, name Recuperação vetorial.
Exemplo: "O que é exigido antes de APIs Web?" -> label Curso,
relationship PRE_REQUISITO, name APIs Web.
Use os três campos vazios se nenhuma entidade específica for mencionada.
A aplicação compila o plano em Cypher de leitura e percorre até dois saltos.
"""
        for attempt in range(2):
            raw = self.client.chat_json(
                model=self.model, system=system, prompt=prompt, schema=PLAN_SCHEMA
            )
            label, relation, name = (
                raw.get(k) for k in ("label", "relationship", "name")
            )
            if label == relation == name == "":
                return None
            if (label, relation) not in QUERY_BY_TYPES or not isinstance(name, str):
                if attempt == 0:
                    prompt += "\nEscolha tipos e relação compatíveis com o esquema."
                    continue
                raise ValueError("Plano Cypher fora do esquema permitido")
            name = name.strip()
            if 2 <= len(name) <= 120 and normalize_text(name) in normalize_text(
                question
            ):
                return CypherPlan(QUERY_BY_TYPES[(label, relation)], name)
            if attempt == 0:
                prompt += (
                    "\nname deve ser uma parte literal da pergunta; não invente nomes."
                )
        # An invalid name must never execute, but hybrid retrieval can continue
        # through the vector/full-text channels and validated neighborhood Cypher.
        return None
