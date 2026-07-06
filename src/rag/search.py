
from __future__ import annotations

"""检索模块：知识库搜索与命中结构体。"""

from dataclasses import dataclass
from pathlib import Path

from src.core.config import settings
from src.rag.embeddings import ensure_store


@dataclass(frozen=True)
class SearchHit:
    """对外暴露的检索命中结构：文本 + 来源文件 + 片段序号 + 距离分数（越小越相似）。"""
    text: str
    source: str
    chunk: int
    distance: float | None = None


def search(query: str, k: int | None = None) -> list[SearchHit]:
    """
    检索最相关的 k 个 chunk；库为空时通过 ensure_store 自动触发首次构建。
    :return: 命中列表，按相似度从高到低
    """
    store = ensure_store()
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
