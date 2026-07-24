from __future__ import annotations

"""基于 LangChain create_agent 的深度思考 Agent。"""

import logging
import time

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.agent.deep.prompts import SYSTEM_PROMPT, build_user_prompt
from src.agent.deep.reflection import ReflectionManager
from src.agent.deep.tool_wrapper import wrap_tools_with_reflection
from src.agent.langchain_result import (
    AgentResult,
    extract_steps_and_sources,
    last_ai_text,
)
from src.tools.rag import get_agent_tools
from src.trace import log_step_ctx


logger = logging.getLogger(__name__)


class DeepThinkingAgent:
    """使用 LangChain create_agent 创建的深度思考 Agent。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_steps: int = 8,
        temperature: float = 0.2,
        timeout: int = 90,
    ) -> None:
        self.max_steps = max_steps
        self.model = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
            streaming=True,
        )
        self.reflection_manager = ReflectionManager()
        self.agent = create_agent(
            model=self.model,
            # 深度思考模式额外包装工具调用，用于发现重复检索或无结果循环。
            tools=wrap_tools_with_reflection(get_agent_tools(), self.reflection_manager),
            system_prompt=SYSTEM_PROMPT,
        )

    def reset_reflection(self) -> None:
        """每一轮深度思考开始前重置反思状态。"""

        self.reflection_manager.reset()

    def run(
        self,
        question: str,
        context_messages: list[dict] | None = None,
        attachment_context: str = "",
        image_urls: list[str] | None = None,
    ) -> AgentResult:
        """运行 Agent，并把 LangChain 消息整理成前端需要的数据结构。"""

        logger.info("深度思考 Agent 开始处理问题：%s", question)
        t0 = time.perf_counter()
        self.reset_reflection()
        # 深度模式把历史对话作为消息传入，同时在 prompt 中提醒模型结合上下文。
        messages = list(context_messages) if context_messages else []
        user_prompt = build_user_prompt(question, context_messages)
        if attachment_context:
            user_prompt = f"{user_prompt}\n\n{attachment_context}"
        content = build_message_content(user_prompt, image_urls or [])
        messages.append({"role": "user", "content": content})
        result = self.agent.invoke(
            {"messages": messages},
            # create_agent 底层由 LangGraph 驱动；这里用递归上限控制最多循环次数。
            config={"recursion_limit": max(6, self.max_steps * 4 + 4)},
        )

        messages = list(result.get("messages", []))
        steps, sources = extract_steps_and_sources(messages)
        answer = last_ai_text(messages) or "暂时没有生成有效回答。"

        log_step_ctx("agent:invoke", ok=True,
                     duration_ms=(time.perf_counter() - t0) * 1000,
                     agent="deep", model=self.model.model_name,
                     steps=len(steps), sources=len(sources), answer_len=len(answer))

        logger.info(
            "深度思考 Agent 完成：steps=%s, sources=%s",
            len(steps),
            len(sources),
        )
        return AgentResult(answer=answer, steps=steps, sources=sources)


def build_message_content(prompt: str, image_urls: list[str]) -> str | list[dict]:
    """根据附件情况构造普通文本或多模态消息。"""

    if not image_urls:
        return prompt

    content: list[dict] = [{"type": "text", "text": prompt}]
    for image_url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content
