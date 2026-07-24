from __future__ import annotations

"""普通问答 Agent 的服务入口。"""

import logging
import re
from dataclasses import asdict

from src.agent.normal.react_agent import NormalAgent
from src.core.config import ModelConfig, settings
from src.memory.service import build_context, save_turn
from src.rag.search import SearchHit
from src.tools.rag import run_rag_search


logger = logging.getLogger(__name__)


CHANNEL_RULES = [
    (
        ("入驻", "注册", "加入企业", "发布需求", "发布供应", "发布项目"),
        "发布与入驻路径",
        "适合完成账号、企业、供需信息发布等基础操作。",
    ),
    (
        ("找项目", "项目机会", "拓展项目"),
        "项目需求频道",
        "适合供应方寻找项目机会，按行业、地区、项目阶段筛选需求。",
    ),
    (
        ("找供应商", "找服务商", "施工", "设计", "咨询", "监理"),
        "项目供应频道",
        "适合需求方寻找咨询、设计、施工、监理等项目服务商。",
    ),
    (
        ("法律", "合同", "合规", "纠纷"),
        "法律服务频道",
        "适合跨境项目合规、合同审核、风险规避和纠纷处理。",
    ),
    (
        ("金融", "融资", "保险", "资金"),
        "金融供应频道",
        "适合项目融资、工程保险、基金投资、跨境结算等金融服务。",
    ),
    (
        ("物流", "运输", "仓储"),
        "物流/仓储供需频道",
        "适合工程物资、设备、原材料运输和仓储托管需求。",
    ),
]


def answer(
    question: str,
    model_config: ModelConfig | None = None,
    conversation_id: str | None = None,
    attachment_context: str = "",
    image_urls: list[str] | None = None,
) -> dict:
    """运行普通 Agent，并返回前端需要的响应结构。"""

    config = model_config or settings.model_config(settings.default_model)
    model_name = config.name
    logger.info("普通问答服务收到问题：model=%s, question=%s, conversation_id=%s", model_name, question, conversation_id)

    # 短期记忆：加载上下文
    cid, context_messages = build_context(conversation_id)

    if not config.api_key:
        logger.warning("普通 Agent 缺少模型 API Key，使用 RAG 兜底回答")
        return _fallback_response(question, model_name, conversation_id=cid)

    agent = NormalAgent(
        model=model_name,
        api_key=config.api_key,
        base_url=config.base_url,
    )

    try:
        result = agent.run(
            question,
            context_messages=context_messages if context_messages else None,
            attachment_context=attachment_context,
            image_urls=image_urls,
        )
    except Exception as exc:
        logger.exception("普通 Agent 运行失败，使用 RAG 兜底回答")
        fallback = _fallback_response(question, model_name, conversation_id=cid)
        fallback["steps"] = [
            {
                "thought": "普通 Agent 调用模型或工具时出现异常，已切换到 RAG 兜底。",
                "action": None,
                "action_input": None,
                "observation": str(exc),
            }
        ]
        return fallback

    # 保存本轮对话，附件上下文一并存入，确保图片分析细节在后续轮次中可见
    save_turn(cid, question, result.answer, attachment_context)

    return {
        "answer": result.answer,
        "recommendations": recommendations(question),
        "sources": result.sources,
        "steps": [asdict(step) for step in result.steps],
        "mode": "normal",
        "model": model_name,
        "conversation_id": cid,
    }


def recommendations(question: str) -> list[dict[str, str]]:
    """根据简单业务关键词生成下一步推荐卡片。"""

    matched = []
    normalized = question.lower()
    for keywords, channel, reason in CHANNEL_RULES:
        if any(keyword.lower() in normalized for keyword in keywords):
            matched.append({"channel": channel, "reason": reason})
    return matched[:3]


def _fallback_response(question: str, model: str | None = None, conversation_id: str = "") -> dict:
    """没有模型或模型失败时，直接展示 RAG 检索证据。"""

    tool_result = run_rag_search(query=question, k=settings.retrieval_k)
    hits = tool_result.hits

    if not hits and (
        tool_result.observation.startswith("知识库目录不存在")
        or tool_result.observation.startswith("知识库检索失败")
    ):
        logger.warning("普通 Agent 无法完成检索：%s", tool_result.observation)
        answer_text = tool_result.observation
    else:
        answer_text = fallback_answer(question, hits)

    return {
        "answer": answer_text,
        "recommendations": recommendations(question),
        "sources": [asdict(hit) for hit in hits],
        "steps": [
            {
                "thought": "使用 RAG 兜底逻辑直接检索知识库。",
                "action": "rag_search",
                "action_input": {"query": question, "k": settings.retrieval_k},
                "observation": tool_result.observation,
            }
        ],
        "mode": "normal",
        "model": model or settings.llm_model,
        "conversation_id": conversation_id,
    }


def fallback_answer(question: str, hits: list[SearchHit]) -> str:
    """没有模型答案时，基于检索结果生成兜底回答。"""

    if not hits:
        return "知识库中未找到足够依据。你可以换一种问法，或补充你的身份、行业、地区和目标。"

    lines = ["我先基于知识库检索到这些相关信息："]
    for index, hit in enumerate(hits[:3], start=1):
        lines.append(
            f"{index}. {compact(hit.text, 220)}（来源：{hit.source}，片段 {hit.chunk}）"
        )

    cards = recommendations(question)
    if cards:
        lines.append("\n建议下一步：")
        lines.extend(f"- 进入{card['channel']}：{card['reason']}" for card in cards)
    else:
        lines.append("\n你可以继续补充：你的身份、业务目标、行业、地区或项目阶段，我可以再帮你缩小路径。")

    return "\n".join(lines)


def compact(text: str, max_len: int) -> str:
    """压缩长文本，避免兜底回答过长。"""

    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[:max_len].rstrip() + "..."
