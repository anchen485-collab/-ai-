from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types

from .config import settings
from .documents import build_chunks
from .embeddings import HashingChineseEmbedding


@dataclass(frozen=True)
class SearchHit:
    text: str
    source: str
    chunk: int
    distance: float | None = None


def get_collection():
    chromadb = _import_chromadb()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = _make_client(chromadb)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=HashingChineseEmbedding(),
        metadata={"description": "全发首页 AI 小助手静态知识库"},
    )


def ingest(source_dir: Path | None = None, reset: bool = True) -> dict[str, int | str]:
    chromadb = _import_chromadb()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = _make_client(chromadb)
    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=HashingChineseEmbedding(),
        metadata={"description": "全发首页 AI 小助手静态知识库"},
    )
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


def _import_chromadb():
    # Chroma 0.5 instantiates its default ONNX embedding during import. We always
    # pass our own embedding function, so the default is never used. Some Windows
    # machines fail while importing onnxruntime, so provide a tiny import-time stub.
    if "onnxruntime" not in sys.modules:
        onnx_stub = types.ModuleType("onnxruntime")
        onnx_stub.get_available_providers = lambda: []
        onnx_stub.get_all_providers = lambda: []
        sys.modules["onnxruntime"] = onnx_stub

    import chromadb

    return chromadb


def _make_client(chromadb):
    chroma_settings = chromadb.config.Settings(
        anonymized_telemetry=False,
        chroma_product_telemetry_impl="src.chroma_noop.NoopTelemetry",
    )
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir), settings=chroma_settings
    )
