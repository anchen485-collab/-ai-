from __future__ import annotations

"""普通 Agent 和深度思考 Agent 共用的 RAG 工具。"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.tools import tool

from src.rag.search import SearchHit, search


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    """工具执行结果，供普通 Agent 和 LangChain 工具复用。"""

    observation: str
    hits: list[SearchHit]


def run_rag_search(query: str, k: int = 5) -> ToolResult:
    """执行知识库检索，返回内部更容易使用的结构。"""

    query = str(query or "").strip()
    k = max(1, min(int(k or 5), 10))
    if not query:
        logger.warning("RAG 工具缺少 query 参数")
        return ToolResult("rag_search 缺少 query 参数。", [])

    logger.info("RAG 检索开始：query=%s, k=%s", query, k)
    try:
        hits = search(query, k=k)
    except FileNotFoundError as exc:
        logger.warning("RAG 检索失败：知识库目录不存在：%s", exc)
        return ToolResult(f"知识库目录不存在，无法检索：{exc}", [])
    except Exception as exc:
        logger.exception("RAG 检索异常：query=%s", query)
        return ToolResult(f"知识库检索失败：{exc}", [])

    logger.info("RAG 检索完成：query=%s, hits=%s", query, len(hits))
    if not hits:
        return ToolResult(f"没有检索到与“{query}”相关的知识库片段。", [])

    lines = [f"检索词：{query}", f"命中数量：{len(hits)}"]
    for index, hit in enumerate(hits, start=1):
        lines.append(
            f"{index}. 来源：{hit.source} / 片段 {hit.chunk} / 距离 {hit.distance}\n"
            f"{hit.text[:600]}"
        )
    return ToolResult("\n\n".join(lines), hits)


@tool
def rag_search(query: str, k: int = 5) -> dict[str, Any]:
    """检索本地 Chroma 知识库，返回相关片段和来源。

    Args:
        query: 检索关键词或用户问题。
        k: 最多返回的片段数量，建议 1 到 10。
    """

    result = run_rag_search(query=query, k=k)
    return {
        "observation": result.observation,
        "sources": serialize_hits(result.hits),
    }


def get_agent_tools() -> list[Any]:
    """返回 Agent 可用工具列表，初始化 LangChain Agent 时直接传入。"""

    return [rag_search]


def serialize_hits(hits: list[SearchHit]) -> list[dict[str, Any]]:
    """把检索结果转换成接口可返回的字典。"""

    return [asdict(hit) for hit in hits]
