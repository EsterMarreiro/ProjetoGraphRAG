"""Orquestração da ingestão dos documentos para o grafo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from licenciagraph.ingestion.canonicalization import EntityCanonicalizer
from licenciagraph.ingestion.curated import PlannedChunk, SourceDocument
from licenciagraph.ingestion.extraction import Extractor
from licenciagraph.ingestion.llm import OpenAIClient
from licenciagraph.ingestion.models import ExtractedEntity, ExtractionResult
from licenciagraph.ingestion.writer import GraphWriter

EMBEDDED_LABELS = frozenset(
    {
        "Atividade",
        "DocumentoExigido",
        "Modalidade",
        "Estudo",
        "Orgao",
        "Norma",
        "Sistema",
        "Grupo",
        "Subgrupo",
    }
)
BATCH_CHUNKS = 24


@dataclass(frozen=True, slots=True)
class IngestionReport:
    documents_found: int
    documents_processed: int
    documents_skipped: int
    chunks_processed: int
    entities_processed: int
    relationships_processed: int

    def as_dict(self) -> dict[str, int]:
        return {
            "documents_found": self.documents_found,
            "documents_processed": self.documents_processed,
            "documents_skipped": self.documents_skipped,
            "chunks_processed": self.chunks_processed,
            "entities_processed": self.entities_processed,
            "relationships_processed": self.relationships_processed,
        }


class IngestionPipeline:
    def __init__(
        self,
        *,
        writer: GraphWriter,
        extractor: Extractor | None,
        llm: OpenAIClient,
        embedding_model: str,
        embedding_dimension: int,
        progress: Any = None,
    ) -> None:
        self.writer = writer
        self.extractor = extractor
        self.llm = llm
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.canonicalizer = EntityCanonicalizer()
        self.progress = progress
        self._embedded: set[str] = set()

    def run(
        self, documents: list[SourceDocument], *, force: bool = False
    ) -> IngestionReport:
        processed = skipped = chunks_count = entities_count = relations_count = 0
        for source in documents:
            if not force and self.writer.is_document_current(source.document):
                skipped += 1
                continue
            self.writer.prepare_document(source.document)
            for start in range(0, len(source.chunks), BATCH_CHUNKS):
                batch = source.chunks[start : start + BATCH_CHUNKS]
                results = [self._extract(item) for item in batch]
                self._write_batch(batch, results)
                chunks_count += len(batch)
                entities_count += sum(len(r.entities) for r in results)
                relations_count += sum(len(r.relationships) for r in results)
                if self.progress:
                    self.progress(
                        source.document.id, start + len(batch), len(source.chunks)
                    )
            self.writer.complete_document(source.document)
            processed += 1
        if processed:
            self.writer.remove_orphans()
        return IngestionReport(
            documents_found=len(documents),
            documents_processed=processed,
            documents_skipped=skipped,
            chunks_processed=chunks_count,
            entities_processed=entities_count,
            relationships_processed=relations_count,
        )

    def _extract(self, item: PlannedChunk) -> ExtractionResult:
        if item.extraction is not None:
            result = item.extraction
        elif self.extractor is not None:
            result = self.extractor.extract(item.chunk.text)
        else:
            result = ExtractionResult()
        for entity in result.entities:
            if entity.canonical_id is None:
                self.canonicalizer.canonicalize(entity)
        return result

    def _write_batch(
        self, batch: list[PlannedChunk], results: list[ExtractionResult]
    ) -> None:
        texts = [item.chunk.text for item in batch]
        pending: list[ExtractedEntity] = []
        for result in results:
            for entity in result.entities:
                key = entity.canonical_id or entity.ref
                if entity.label in EMBEDDED_LABELS and key not in self._embedded:
                    self._embedded.add(key)
                    pending.append(entity)
        vectors = self.llm.embed(
            model=self.embedding_model,
            texts=[*texts, *(_entity_text(e) for e in pending)],
            dimensions=self.embedding_dimension,
        )
        entity_vectors = {
            (e.canonical_id or e.ref): vector
            for e, vector in zip(pending, vectors[len(texts) :], strict=True)
        }
        for item, vector, result in zip(batch, vectors, results, strict=False):
            self.writer.upsert_chunk(item.chunk, vector)
            by_ref = {entity.ref: entity for entity in result.entities}
            for entity in result.entities:
                key = entity.canonical_id or entity.ref
                self.writer.upsert_entity(entity, entity_vectors.pop(key, None))
                self.writer.link_mention(item.chunk.id, entity)
            for relationship in result.relationships:
                self.writer.upsert_relationship(
                    relationship, by_ref, chunk_id=item.chunk.id
                )


def _entity_text(entity: ExtractedEntity) -> str:
    properties = entity.properties
    context = [
        str(properties[key])
        for key in ("grupo", "subgrupo", "categoria", "descricao")
        if properties.get(key) and str(properties[key]) != entity.name
    ]
    return ". ".join([entity.name, *context[:2]])
