from __future__ import annotations

"""基于 LangChain create_agent 的普通问答 Agent。"""

import logging

from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi

from src.agent.langchain_result import (
    AgentResult,
    extract_steps_and_sources,
    last_ai_text,
)
from src.agent.normal.prompts import SYSTEM_PROMPT
from src.tools.rag import get_agent_tools


logger = logging.getLogger(__name__)


class NormalAgent:
    """使用通义模型和共享 RAG 工具的普通 Agent。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        max_steps: int = 4,
    ) -> None:
        self.max_steps = max_steps
        self.model = ChatTongyi(
            model=model,
            api_key=api_key,
            temperature=0.2,
        )
        self.agent = create_agent(
            model=self.model,
            tools=get_agent_tools(),
            system_prompt=SYSTEM_PROMPT,
        )

    def run(self, question: str) -> AgentResult:
        """运行普通 Agent，并返回统一结果结构。"""

        logger.info("普通 Agent 开始处理问题：%s", question)
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": max(6, self.max_steps * 4 + 4)},
        )

        messages = list(result.get("messages", []))
        steps, sources = extract_steps_and_sources(messages)
        answer = last_ai_text(messages) or "暂时没有生成有效回答。"

        logger.info(
            "普通 Agent 完成：steps=%s, sources=%s",
            len(steps),
            len(sources),
        )
        return AgentResult(answer=answer, steps=steps, sources=sources)
