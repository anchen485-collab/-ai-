from __future__ import annotations

"""问答 Agent 编排层。

这个模块不直接处理 HTTP，也不直接解析 docx。
它负责把“用户问题”组织成完整回答：
1. 从 Chroma 检索知识库片段。
2. 按关键词给出频道/路径推荐。
3. 有 DashScope Key 时调用通义千问生成自然语言回答。
4. 没有 Key 或模型失败时，返回可用的检索兜底回答。
"""

import os
import re
from dataclasses import asdict

from .config import settings
from .kb import SearchHit, search

# 一期先用轻量关键词规则做频道推荐。
# 这些规则不替代 RAG，只负责给前端展示“下一步建议”卡片。
CHANNEL_RULES = [
    (("入驻", "注册", "加入企业", "发布需求", "发布供应", "发布项目"), "发布与入驻路径",
     "适合完成账号、企业、供求信息发布等基础操作。"),
    (("找项目", "项目机会", "拓展项目"), "项目需求频道",
     "适合供应方寻找项目机会，按行业、地区、项目阶段筛选需求。"),
    (("找供应商", "找服务商", "施工", "设计", "咨询", "监理"), "项目供应频道",
     "适合需求方寻找咨询、设计、施工、监理等项目服务商。"),
    (("法律", "合同", "合规", "纠纷"), "法律服务频道", "适合跨境项目合规、合同审核、风险规避和纠纷处理。"),
    (("金融", "融资", "保险", "资金"), "金融供应频道", "适合项目融资、工程保险、基金投资、跨境结算等金融服务。"),
    (("物流", "运输", "仓储"), "物流/仓储供需频道", "适合工程物资、设备、原材料运输和仓储托管需求。"),
]


def detect_recommendations(question: str) -> list[dict[str, str]]:
    """根据用户问题匹配推荐频道或操作路径。"""

    normalized = question.lower()
    cards: list[dict[str, str]] = []
    for keywords, channel, reason in CHANNEL_RULES:
        if any(keyword.lower() in normalized for keyword in keywords):
            cards.append({"channel": channel, "reason": reason})
    return cards[:3]


def answer(question: str) -> dict:
    """Agent 主入口。

    返回结构固定为 answer/recommendations/sources，方便前端直接渲染。
    """

    hits = search(question)
    llm_answer = _llm_answer(question, hits)
    if not llm_answer:
        llm_answer = _fallback_answer(question, hits)

    return {
        "answer": llm_answer,
        "recommendations": detect_recommendations(question),
        "sources": [asdict(hit) for hit in hits],
    }


def _llm_answer(question: str, hits: list[SearchHit]) -> str | None:
    """尝试调用通义千问生成答案。

    如果没有配置 DASHSCOPE_API_KEY，或者模型调用失败，返回 None，
    由 _fallback_answer 接管，保证系统不因为模型不可用而整体失败。
    """

    if not os.getenv("DASHSCOPE_API_KEY"):
        return None
    try:
        from langchain_community.chat_models import ChatTongyi
    except Exception:
        return None

    context = "\n\n".join(
        f"【来源：{hit.source} / 片段 {hit.chunk}】\n{hit.text}" for hit in hits
    )
    # Prompt 明确要求“只基于知识库回答”，降低编造不存在企业/项目的风险。
    prompt = f"""你是“全发首页 AI 小助手”，负责平台导览、知识库问答和频道推荐。

请只基于下方知识库内容回答用户问题，不要编造不存在的企业、项目、资质、案例或交易结果。
如果知识库没有依据，请明确说明“知识库中未找到足够依据”，并给出可追问的问题或转人工建议。
回答要简洁、可操作，必要时给出下一步路径。

知识库内容：
{context}

用户问题：{question}
"""
    try:
        llm = ChatTongyi(model=settings.llm_model, temperature=0.2)
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()
    except Exception:
        return None


def _fallback_answer(question: str, hits: list[SearchHit]) -> str:
    """无模型时的兜底回答。

    直接展示检索片段和来源，牺牲表达自然度，换取可验证、可运行。
    """

    if not hits:
        return "知识库中未找到足够依据。你可以换一种问法，或补充你的身份、行业、地区和目标。"

    best = hits[:3]
    lines = ["我先基于知识库检索到这些相关信息："]
    for index, hit in enumerate(best, start=1):
        snippet = _compact(hit.text, 220)
        lines.append(f"{index}. {snippet}（来源：{hit.source}，片段 {hit.chunk}）")

    cards = detect_recommendations(question)
    if cards:
        lines.append("\n建议下一步：")
        for card in cards:
            lines.append(f"- 进入{card['channel']}：{card['reason']}")
    else:
        lines.append("\n你可以继续补充：你的身份、业务目标、行业、地区或项目阶段，我可以再帮你缩小路径。")
    return "\n".join(lines)


def _compact(text: str, max_len: int) -> str:
    """压缩长片段，避免兜底回答过长。"""

    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip() + "..."
