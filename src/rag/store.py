from __future__ import annotations

"""Chroma vector-store wrapper used by the rest of the project."""

import sys
import types
from dataclasses import dataclass
from pathlib import Path

from src.core.config import settings
from src.rag.documents import build_chunks
from src.rag.embeddings import HashingChineseEmbedding


@dataclass(frozen=True)
class SearchHit:
    text: str
    source: str
    chunk: int
    distance: float | None = None


def get_collection():
    chromadb = import_chromadb()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chroma_client(chromadb)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=HashingChineseEmbedding(),
        metadata={"description": "全发首页 AI 小助手静态知识库"},
    )


def ingest(source_dir: Path | None = None, reset: bool = True) -> dict[str, int | str]:
    """Rebuild the Chroma collection from docx knowledge files."""

    chromadb = import_chromadb()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chroma_client(chromadb)

    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:
            pass

    collection = get_collection()
    chunks = build_chunks(source_dir)
    if chunks:
        collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

    return {
        "collection": settings.chroma_collection,
        "documents": len({chunk.metadata["source"] for chunk in chunks}),
        "chunks": len(chunks),
        "path": str(settings.chroma_dir),
    }


def search(query: str, k: int | None = None) -> list[SearchHit]:
    """Return the most relevant knowledge chunks for a query."""

    collection = get_collection()
    if collection.count() == 0:
        ingest(reset=False)

    result = collection.query(
        query_texts=[query],
        n_results=k or settings.retrieval_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    hits: list[SearchHit] = []
    for doc, metadata, distance in zip(docs, metadatas, distances):
        hits.append(
            SearchHit(
                text=doc,
                source=str(metadata.get("source", "未知来源")),
                chunk=int(metadata.get("chunk", 0) or 0),
                distance=float(distance) if distance is not None else None,
            )
        )
    return hits


def import_chromadb():
    """Import Chroma while bypassing its unused default ONNX embedding."""

    if "onnxruntime" not in sys.modules:
        stub = types.ModuleType("onnxruntime")
        stub.get_available_providers = lambda: []
        stub.get_all_providers = lambda: []
        sys.modules["onnxruntime"] = stub

    import chromadb

    return chromadb


def chroma_client(chromadb):
    """Create a persistent Chroma client with local-friendly settings."""

    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=chromadb.config.Settings(
            anonymized_telemetry=False,
            chroma_product_telemetry_impl="src.rag.chroma_noop.NoopTelemetry",
        ),
    )
