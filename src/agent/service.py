from __future__ import annotations

"""助手问答编排：检索知识库 -> 生成回答 -> 给出频道推荐。"""

import os
import re
from dataclasses import asdict

from src.core.config import settings
from src.rag.embeddings import SearchHit, search


# 关键词 -> 频道的映射规则，用于在回答下方给出“下一步操作卡片”。
# 每条规则：(触发关键词元组, 频道名, 推荐理由)
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


def answer(question: str) -> dict:
    """主入口：检索 -> 调 LLM（可选）-> 拼装最终响应。"""

    # 1. 先检索知识库，命中结果会同时作为 LLM 的上下文和兜底回答的来源
    hits = search(question)
    # 2. 优先用 LLM 生成自然语言回答；未配置 Key 或调用失败时走兜底
    text = llm_answer(question, hits) or fallback_answer(question, hits)

    return {
        "answer": text,
        "recommendations": recommendations(question),
        "sources": [asdict(hit) for hit in hits],
    }


def recommendations(question: str) -> list[dict[str, str]]:
    """基于关键词命中匹配频道卡片，最多返回 3 条避免刷屏。"""

    matched = []
    # 转小写后做包含匹配，保证大小写不敏感
    normalized = question.lower()
    for keywords, channel, reason in CHANNEL_RULES:
        if any(keyword.lower() in normalized for keyword in keywords):
            matched.append({"channel": channel, "reason": reason})
    return matched[:3]


def llm_answer(question: str, hits: list[SearchHit]) -> str | None:
    """在配置了 DASHSCOPE_API_KEY 时调用通义千问生成回答，否则返回 None。"""

    # 没有配置 Key 就跳过 LLM，保持项目可独立运行
    if not os.getenv("DASHSCOPE_API_KEY"):
        return None

    try:
        # 延迟导入：未配置 Key 的环境无需安装 langchain-community
        from langchain_community.chat_models import ChatTongyi

        # 把检索到的片段按来源标注后拼成上下文，方便模型引用
        context = "\n\n".join(
            f"【来源：{hit.source} / 片段 {hit.chunk}】\n{hit.text}" for hit in hits
        )
        prompt = f"""你是“全发首页 AI 小助手”，负责平台导览、知识库问答和频道推荐。

请只基于下方知识库内容回答用户问题，不要编造不存在的企业、项目、资质、案例或交易结果。
如果知识库没有依据，请明确说明“知识库中未找到足够依据”，并给出可追问的问题或转人工建议。
回答要简洁、可操作，必要时给出下一步路径。

知识库内容：
{context}

用户问题：{question}
"""
        # temperature 设低以减少胡编，保持回答稳定可控
        response = ChatTongyi(model=settings.llm_model, temperature=0.2).invoke(prompt)
        return response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
    except Exception:
        # 任何异常都降级到兜底回答，避免一次外部调用失败导致整个接口 500
        return None


def fallback_answer(question: str, hits: list[SearchHit]) -> str:
    """无 LLM 可用时的兜底回答：直接展示检索证据 + 推荐频道。"""

    if not hits:
        return "知识库中未找到足够依据。你可以换一种问法，或补充你的身份、行业、地区和目标。"

    # 仅展示前 3 条证据，避免响应过长
    lines = ["我先基于知识库检索到这些相关信息："]
    for index, hit in enumerate(hits[:3], start=1):
        lines.append(f"{index}. {compact(hit.text, 220)}（来源：{hit.source}，片段 {hit.chunk}）")

    # 拼接下一步推荐；无匹配时给出引导性问题
    cards = recommendations(question)
    if cards:
        lines.append("\n建议下一步：")
        lines.extend(f"- 进入{card['channel']}：{card['reason']}" for card in cards)
    else:
        lines.append("\n你可以继续补充：你的身份、业务目标、行业、地区或项目阶段，我可以再帮你缩小路径。")

    return "\n".join(lines)


def compact(text: str, max_len: int) -> str:
    """压缩空白并截断到 max_len，避免单条证据过长。"""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[:max_len].rstrip() + "..."
