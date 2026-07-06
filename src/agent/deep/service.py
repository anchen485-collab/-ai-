from __future__ import annotations

"""FastAPI 可直接调用的深度思考 Agent 服务。"""

import logging
from dataclasses import asdict

from src.agent.deep.react_agent import DeepThinkingAgent
from src.agent.normal.service import recommendations
from src.core.config import settings


logger = logging.getLogger(__name__)


def deep_answer(question: str) -> dict:
    """运行深度思考 Agent，并返回和普通问答兼容的响应结构。"""

    logger.info("深度思考服务收到问题：%s", question)
    if not settings.openai_compatible_api_key:
        logger.warning("深度思考模式缺少 API Key")
        return {
            "answer": (
                "深度思考模式需要先配置 API Key。\n"
                "请在 .env 中设置 OPENAI_COMPATIBLE_API_KEY，"
                "如果使用 DeepSeek，建议同时设置 "
                "OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com "
                "和 DEEP_AGENT_MODEL。"
            ),
            "recommendations": recommendations(question),
            "sources": [],
            "steps": [
                {
                    "thought": "启动前检查配置，发现深度思考模型缺少 API Key。",
                    "action": None,
                    "action_input": None,
                    "observation": "未调用模型。",
                }
            ],
            "mode": "deep",
        }

    agent = DeepThinkingAgent(
        api_key=settings.openai_compatible_api_key,
        base_url=settings.openai_compatible_base_url,
        model=settings.deep_agent_model,
        max_steps=settings.deep_agent_max_steps,
        temperature=settings.deep_agent_temperature,
        timeout=settings.deep_agent_timeout,
    )

    try:
        result = agent.run(question)
    except Exception as exc:
        logger.exception("深度思考模式运行失败")
        return {
            "answer": f"深度思考模式运行失败：{exc}",
            "recommendations": recommendations(question),
            "sources": [],
            "steps": [
                {
                    "thought": "调用深度思考模型或工具时出现异常。",
                    "action": None,
                    "action_input": None,
                    "observation": str(exc),
                }
            ],
            "mode": "deep",
        }

    logger.info(
        "深度思考服务完成回答：steps=%s, sources=%s",
        len(result.steps),
        len(result.sources),
    )
    return {
        "answer": result.answer,
        "recommendations": recommendations(question),
        "sources": result.sources,
        "steps": [asdict(step) for step in result.steps],
        "mode": "deep",
    }
