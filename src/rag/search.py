from __future__ import annotations

"""检索模块：MMR 粗召回 → DashScope Rerank 精排 → 阈值过滤。"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from src.core.config import settings
from src.rag.embeddings import _ensure_store as ensure_store

MMR_FETCH_MULTIPLIER = 4
MMR_LAMBDA = 0.6

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _retry_rerank(max_retries: int = 2, base_delay: float = 0.5) -> Callable[[F], F]:
    """带指数退避的重试装饰器，用于 Rerank API 瞬时故障恢复。"""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "Rerank API 第 %d/%d 次失败，%ss 后重试：%s",
                            attempt + 1, max_retries + 1, delay, exc,
                        )
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


@dataclass(frozen=True)
class SearchHit:
    """对外暴露的检索命中结构：文本 + 来源文件 + 片段序号 + 可选分数。"""
    text: str
    source: str
    chunk: int
    distance: float | None = None
    rerank_score: float | None = None


def serialize_hits(hits: list[SearchHit]) -> list[dict[str, Any]]:
    """将检索结果转为可序列化的字典列表。"""
    from dataclasses import asdict
    return [asdict(hit) for hit in hits]


def search(query: str, k: int | None = None) -> list[SearchHit]:
    """MMR 粗召回 → rerank 精排 → 阈值过滤 → 返回 top-K。"""
    k = k or settings.retrieval_k
    candidates = _mmr_recall(query)
    if not candidates:
        return []

    ranked = _rerank(query, candidates, top_n=k)
    if ranked is None:
        return candidates[:k]

    filtered = [hit for hit in ranked if (hit.rerank_score or 0) >= settings.rerank_threshold]
    if not filtered:
        logger.warning("Rerank 后无结果通过阈值 %.2f，回退到 MMR top-K", settings.rerank_threshold)
        return candidates[:k]

    return filtered[:k]


def _mmr_recall(query: str) -> list[SearchHit]:
    """MMR 召回 candidate pool，兼顾相似度和多样性。"""
    store = ensure_store()
    results = store.max_marginal_relevance_search(
        query,
        k=settings.rerank_fetch_k,
        fetch_k=settings.rerank_fetch_k * MMR_FETCH_MULTIPLIER,
        lambda_mult=MMR_LAMBDA,
    )
    hits: list[SearchHit] = []
    for doc in results:
        source_path = doc.metadata.get("source_path", "")
        hits.append(
            SearchHit(
                text=doc.page_content,
                source=Path(source_path).name if source_path else "未知来源",
                chunk=int(doc.metadata.get("chunk", 0) or 0),
            )
        )
    return hits


def _rerank(query: str, candidates: list[SearchHit], top_n: int) -> list[SearchHit] | None:
    """调用 DashScope Rerank API 精排；不可用时返回 None。"""
    if not settings.dashscope_api_key:
        logger.warning("未配置 DASHSCOPE_API_KEY，跳过 rerank")
        return None

    @_retry_rerank(max_retries=2, base_delay=0.5)
    def _call_rerank_api() -> Any:
        from dashscope.rerank import TextReRank

        return TextReRank.call(
            model=settings.rerank_model,
            query=query,
            documents=[hit.text for hit in candidates],
            top_n=top_n,
            return_documents=False,
            api_key=settings.dashscope_api_key,
        )

    try:
        response = _call_rerank_api()
    except Exception as exc:
        logger.warning("Rerank API 重试后仍失败，回退 MMR: %s", exc)
        return None

    output = getattr(response, "output", None)
    results = output.get("results", []) if output else []
    if not results:
        return None

    ranked: list[SearchHit] = []
    for item in sorted(results, key=lambda r: r.get("index", 0)):
        idx = item.get("index", 0)
        if idx < len(candidates):
            hit = candidates[idx]
            ranked.append(
                SearchHit(
                    text=hit.text,
                    source=hit.source,
                    chunk=hit.chunk,
                    distance=hit.distance,
                    rerank_score=item.get("relevance_score"),
                )
            )

    ranked.sort(key=lambda h: h.rerank_score or 0, reverse=True)
    return ranked
