from __future__ import annotations

"""基于 LangChain create_agent 的普通问答 Agent。"""

import logging
import time

from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI

from src.agent.langchain_result import (
    AgentResult,
    extract_steps_and_sources,
    last_ai_text,
)
from src.agent.normal.prompts import SYSTEM_PROMPT
from src.tools.rag import get_agent_tools
from src.trace import log_step_ctx


logger = logging.getLogger(__name__)


class NormalAgent:
    """使用通义模型和共享 RAG 工具的普通 Agent。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "",
        max_steps: int = 4,
    ) -> None:
        self.max_steps = max_steps
        if base_url:
            self.model = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.2,
                streaming=True,
            )
        else:
            self.model = ChatTongyi(
                model=model,
                api_key=api_key,
                temperature=0.2,
                streaming=True,
            )
        self.agent = create_agent(
            model=self.model,
            tools=get_agent_tools(),
            system_prompt=SYSTEM_PROMPT,
        )

    def run(
        self,
        question: str,
        context_messages: list[dict] | None = None,
        attachment_context: str = "",
        image_urls: list[str] | None = None,
    ) -> AgentResult:
        """运行普通 Agent，并返回统一结果结构。"""

        logger.info("普通 Agent 开始处理问题：%s", question)
        t0 = time.perf_counter()
        # 先放入历史对话，再追加本轮用户消息。
        messages = list(context_messages) if context_messages else []
        content = build_message_content(question, attachment_context, image_urls or [])
        messages.append({"role": "user", "content": content})
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": max(6, self.max_steps * 4 + 4)},
        )

        messages = list(result.get("messages", []))
        steps, sources = extract_steps_and_sources(messages)
        answer = last_ai_text(messages) or "暂时没有生成有效回答。"

        log_step_ctx("agent:invoke", ok=True,
                     duration_ms=(time.perf_counter() - t0) * 1000,
                     agent="normal", model=self.model.model_name,
                     steps=len(steps), sources=len(sources), answer_len=len(answer))

        logger.info(
            "普通 Agent 完成：steps=%s, sources=%s",
            len(steps),
            len(sources),
        )
        return AgentResult(answer=answer, steps=steps, sources=sources)


def build_message_content(
    question: str,
    attachment_context: str,
    image_urls: list[str],
) -> str | list[dict]:
    """根据附件情况构造普通文本或多模态消息。"""

    text = question if not attachment_context else f"{question}\n\n{attachment_context}"
    if not image_urls:
        return text

    content: list[dict] = [{"type": "text", "text": text}]
    for image_url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content
