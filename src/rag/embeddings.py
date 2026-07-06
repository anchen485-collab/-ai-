from __future__ import annotations

# src/rag/embeddings.py
# 职责：向量库管理 + 入库时的自动向量化
# 说明：
# - DashScopeEmbeddings 作为 embedding 函数注入 Chroma，add_documents 时自动向量化
# - 一个 file_hash 对应一个 collection（持久化目录也按 hash 隔离），知识库内容变更即新建
# - 不再有"单独的向量化步骤"——调用方只管给 chunks，向量化由 Chroma 内部完成

"""向量库与向量化：负责加载已有 Chroma 库或用 chunks 新建（新建时自动向量化）。

对外暴露：
- VectorStoreManager: 向量库管理器，封装 load_or_build
- SearchHit: 检索命中结构体
- ingest() / search(): 模块级便捷函数
"""

import logging
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

if "onnxruntime" not in sys.modules:
    # Chroma 导入时会初始化默认 ONNX embedding，但本项目实际使用 DashScopeEmbeddings。
    # 某些 Windows 环境中 onnxruntime DLL 会初始化失败，这里用最小 stub 避免导入阶段崩溃。
    stub = types.ModuleType("onnxruntime")
    stub.get_available_providers = lambda: []
    stub.get_all_providers = lambda: []
    sys.modules["onnxruntime"] = stub

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from chromadb.config import Settings as ChromaSettings

from src.core.config import Settings, settings
from src.rag.documents import DocxIngestor


# =============================================================================
# 数据结构
# =============================================================================

@dataclass(frozen=True)
class SearchHit:
    """对外暴露的检索命中结构：文本 + 来源文件 + 片段序号 + 距离分数（越小越相似）。"""
    text: str
    source: str
    chunk: int
    distance: float | None = None


# =============================================================================
# 向量库管理器
# =============================================================================

class VectorStoreManager:
    """向量库管理器：负责加载已有库或新建并向量化入库。

    典型用法：
        manager = VectorStoreManager(config, embeddings)
        store = manager.load_or_build(file_hash, chunks)
        # store 已经包含向量化后的数据，可直接 similarity_search
    """

    def __init__(self, config: Settings, embeddings: DashScopeEmbeddings) -> None:
        """
        :param config: 全局配置（包含 chroma_dir）
        :param embeddings: Embedding 模型实例（已绑定 DASHSCOPE_API_KEY）
        """
        self.__config = config
        self._embeddings = embeddings

    def load_or_build(
        self,
        file_hash: str,
        chunks: Optional[List[Document]] = None,
    ) -> Chroma:
        """
        加载已有向量库；不存在时用 chunks 新建（add_documents 时自动向量化）
        :param file_hash: 知识库内容哈希，决定 collection 名与持久化目录（不同 hash 互不影响）
        :param chunks: 切分后的 Document 列表（新建时必须提供；已有数据时忽略）
        :return: Chroma 向量库实例
        """
        persist_dir = os.path.join(str(self.__config.chroma_dir), f"kb_{file_hash}")
        collection_name = f"kb_{file_hash}"

        # 传入 embedding_function：Chroma 在 add_documents / similarity_search 时自动调用
        store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
            client_settings=ChromaSettings(anonymized_telemetry=False),
            collection_metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )

        # 已有数据则直接复用，避免重复向量化
        if store._collection.count() > 0:
            return store

        # 库为空且未提供 chunks：无法建库
        if not chunks:
            raise ValueError("向量库不存在且未提供 chunks，无法建库")

        # add_documents 时 Chroma 内部自动调用 embedding_function 完成向量化
        store.add_documents(
            documents=chunks,
            ids=[f"id-{idx}" for idx in range(1, len(chunks) + 1)],
        )
        return store


# =============================================================================
# 模块级便捷函数：屏蔽 VectorStoreManager 与 DocxIngestor 的协作细节
# =============================================================================

_default_manager: Optional[VectorStoreManager] = None


def _manager() -> VectorStoreManager:
    """惰性单例：避免导入时立刻创建 Chroma 客户端。"""
    global _default_manager
    if _default_manager is None:
        _default_manager = VectorStoreManager(settings, DashScopeEmbeddings())
    return _default_manager


def _ensure_store() -> Chroma:
    """加载或构建当前 kb_hash 对应的向量库。

    策略：先尝试用 None 复用已有库；不存在时才加载 docx 并切分。
    这样知识库未变更时不会重复做 IO / 切分。
    """
    ingestor = DocxIngestor(settings)
    file_hash = ingestor.kb_hash()
    manager = _manager()
    try:
        # 已有数据：直接复用
        return manager.load_or_build(file_hash, None)
    except ValueError:
        # 库不存在：加载并切分 docx，新建 collection（add_documents 时自动向量化）
        chunks = ingestor.ingest()
        return manager.load_or_build(file_hash, chunks)


def ingest() -> dict:
    """
    知识库入库：按当前目录内容哈希决定复用还是新建。
    - kb_hash 未变 → 直接复用已有 collection
    - kb_hash 变化 → 切分 docx，新建 collection 并自动向量化
    """
    store = _ensure_store()
    return {
        "collection": store._collection.name,
        "chunks": store._collection.count(),
        "path": str(store._persist_directory),
    }


def ingest_new() -> dict:
    """
    增量入库：在新模型下与 ingest() 等价。
    kb_hash 不变则复用，变化则新建，无需单独的增量逻辑。
    """
    return ingest()


def search(query: str, k: int | None = None) -> list[SearchHit]:
    """
    检索最相关的 k 个 chunk；库为空时通过 _ensure_store 自动触发首次构建。
    :return: 命中列表，按相似度从高到低
    """
    store = _ensure_store()
    # similarity_search_with_score 返回 (Document, distance) 元组列表
    results = store.similarity_search_with_score(
        query, k=k or settings.retrieval_k
    )
    hits: list[SearchHit] = []
    for doc, score in results:
        source_path = doc.metadata.get("source_path", "")
        hits.append(
            SearchHit(
                text=doc.page_content,
                source=Path(source_path).name if source_path else "未知来源",
                chunk=int(doc.metadata.get("chunk", 0) or 0),
                distance=float(score),
            )
        )
    return hits
