from __future__ import annotations

"""Chroma 知识库访问层。

这个模块是项目里唯一直接接触 Chroma 的地方。
其他模块只需要调用 ingest() 或 search()，不需要关心 Chroma 初始化细节。
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import types

from .config import settings
from .documents import build_chunks
from .embeddings import HashingChineseEmbedding


@dataclass(frozen=True)
class SearchHit:
    """一次检索命中的知识片段。"""

    text: str
    source: str
    chunk: int
    distance: float | None = None


def get_collection():
    """获取或创建 Chroma collection。"""

    chromadb = _import_chromadb()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = _make_client(chromadb)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=HashingChineseEmbedding(),
        metadata={"description": "全发首页 AI 小助手静态知识库"},
    )


def ingest(source_dir: Path | None = None, reset: bool = True) -> dict[str, int | str]:
    """把知识库 docx 解析后写入 Chroma。

    reset=True 时会先删除旧 collection，适合重新构建静态知识库。
    """

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
        # Chroma 会调用 collection 的 embedding_function，把 documents 转成向量。
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
    """检索和 query 最相关的知识库片段。"""

    collection = get_collection()
    if collection.count() == 0:
        # 如果用户还没手动跑入库脚本，这里做一次懒加载入库。
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
        # 将 Chroma 的嵌套返回结构转换为前端和 Agent 更好用的 SearchHit。
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
    """创建 Chroma 持久化客户端。

    这里禁用匿名遥测，并接入 NoopTelemetry，保持本地日志干净。
    """

    chroma_settings = chromadb.config.Settings(
        anonymized_telemetry=False,
        chroma_product_telemetry_impl="src.chroma_noop.NoopTelemetry",
    )
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir), settings=chroma_settings
    )
