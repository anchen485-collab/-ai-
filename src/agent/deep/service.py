from __future__ import annotations

"""FastAPI 可直接调用的深度思考 Agent 服务。"""

import logging
from dataclasses import asdict

from src.agent.deep.react_agent import DeepThinkingAgent
from src.agent.normal.service import recommendations
from src.core.config import ModelConfig, settings
from src.memory.service import build_context, save_turn


logger = logging.getLogger(__name__)


def deep_answer(
    question: str,
    model_config: ModelConfig | None = None,
    conversation_id: str | None = None,
    attachment_context: str = "",
    image_urls: list[str] | None = None,
) -> dict:
    """运行深度思考 Agent，并返回和普通问答兼容的响应结构。"""

    config = model_config or settings.model_config(settings.default_model)
    model_name = config.name
    logger.info("深度思考服务收到问题：model=%s, question=%s, conversation_id=%s", model_name, question, conversation_id)

    # 短期记忆：加载上下文
    cid, context_messages = build_context(conversation_id)

    if not config.api_key:
        logger.warning("深度思考模式缺少 API Key")
        return {
            "answer": (
                f"模型 {model_name} 缺少 API Key。\n"
                "请在 .env 中设置对应厂商的 API Key，例如 DASHSCOPE_API_KEY 或 OPENAI_COMPATIBLE_API_KEY。"
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
            "model": model_name,
            "conversation_id": cid,
        }

    agent = DeepThinkingAgent(
        api_key=config.api_key,
        base_url=config.base_url,
        model=model_name,
        max_steps=settings.deep_agent_max_steps,
        temperature=settings.deep_agent_temperature,
        timeout=settings.deep_agent_timeout,
    )

    try:
        result = agent.run(
            question,
            context_messages=context_messages if context_messages else None,
            attachment_context=attachment_context,
            image_urls=image_urls,
        )
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
            "model": model_name,
            "conversation_id": cid,
        }

    # 保存本轮对话，附件上下文一并存入，确保图片分析细节在后续轮次中可见
    save_turn(cid, question, result.answer, attachment_context)

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
        "model": model_name,
        "conversation_id": cid,
    }
