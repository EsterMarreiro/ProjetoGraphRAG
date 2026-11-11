"""Orquestracao da ingestao de TXT para o grafo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from trilhagraph.ingestion.canonicalization import EntityCanonicalizer
from trilhagraph.ingestion.chunking import split_document
from trilhagraph.ingestion.loader import load_documents
from trilhagraph.ingestion.models import ExtractionResult
from trilhagraph.ingestion.ollama import OllamaClient
from trilhagraph.ingestion.writer import GraphWriter


class Extractor(Protocol):
    def extract(self, text: str) -> ExtractionResult: ...


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
        extractor: Extractor,
        ollama: OllamaClient,
        embedding_model: str,
        embedding_dimension: int,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        self.writer = writer
        self.extractor = extractor
        self.ollama = ollama
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.canonicalizer = EntityCanonicalizer()

    def run(self, data_dir: Path, *, force: bool = False) -> IngestionReport:
        documents = load_documents(data_dir)
        processed = skipped = chunks_count = entities_count = relations_count = 0

        for document in documents:
            if not force and self.writer.is_document_current(document):
                skipped += 1
                continue

            chunks = split_document(
                document,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            self.writer.prepare_document(document)
            for chunk in chunks:
                extraction = self.extractor.extract(chunk.text)
                for entity in extraction.entities:
                    self.canonicalizer.canonicalize(entity)

                entity_texts = [
                    _entity_embedding_text(entity.name, entity.properties)
                    for entity in extraction.entities
                ]
                embeddings = self.ollama.embed(
                    model=self.embedding_model,
                    texts=[chunk.text, *entity_texts],
                    dimensions=self.embedding_dimension,
                )
                self.writer.upsert_chunk(chunk, embeddings[0])

                entities_by_ref = {entity.ref: entity for entity in extraction.entities}
                for entity, embedding in zip(
                    extraction.entities, embeddings[1:], strict=True
                ):
                    self.writer.upsert_entity(entity, embedding)
                    self.writer.link_mention(chunk.id, entity)
                for relationship in extraction.relationships:
                    self.writer.upsert_relationship(
                        relationship,
                        entities_by_ref,
                        chunk_id=chunk.id,
                    )

                chunks_count += 1
                entities_count += len(extraction.entities)
                relations_count += len(extraction.relationships)
            self.writer.complete_document(document)
            processed += 1

        return IngestionReport(
            documents_found=len(documents),
            documents_processed=processed,
            documents_skipped=skipped,
            chunks_processed=chunks_count,
            entities_processed=entities_count,
            relationships_processed=relations_count,
        )


def _entity_embedding_text(name: str, properties: dict[str, object]) -> str:
    description = properties.get("descricao")
    return f"{name}. {description}" if description else name
