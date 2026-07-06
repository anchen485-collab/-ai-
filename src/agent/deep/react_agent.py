from __future__ import annotations

"""基于 LangChain create_agent 的深度思考 Agent。"""

import ast
import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.agent.deep.prompts import SYSTEM_PROMPT, build_user_prompt
from src.tools.rag import get_agent_tools


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentStep:
    """记录一次工具调用过程，方便前端展示深度思考轨迹。"""

    thought: str
    action: str | None = None
    action_input: Any = None
    observation: str | None = None


@dataclass(frozen=True)
class AgentResult:
    """深度思考 Agent 的最终结果。"""

    answer: str
    steps: list[AgentStep]
    sources: list[dict[str, Any]]


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
        )
        self.agent = create_agent(
            model=self.model,
            tools=get_agent_tools(),
            system_prompt=SYSTEM_PROMPT,
        )

    def run(self, question: str) -> AgentResult:
        """运行 Agent，并把 LangChain 消息整理成前端需要的数据结构。"""

        logger.info("深度思考 Agent 开始处理问题：%s", question)
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": build_user_prompt(question)}]},
            # create_agent 底层由 LangGraph 驱动；这里用递归上限控制最多循环次数。
            config={"recursion_limit": max(6, self.max_steps * 4 + 4)},
        )

        messages = list(result.get("messages", []))
        steps, sources = _extract_steps_and_sources(messages)
        answer = _last_ai_text(messages) or "暂时没有生成有效回答。"

        logger.info(
            "深度思考 Agent 完成：steps=%s, sources=%s",
            len(steps),
            len(sources),
        )
        return AgentResult(answer=answer, steps=steps, sources=sources)


def _extract_steps_and_sources(messages: list[Any]) -> tuple[list[AgentStep], list[dict[str, Any]]]:
    """从 LangChain 消息中提取工具调用步骤和 RAG 来源。"""

    steps: list[AgentStep] = []
    sources: list[dict[str, Any]] = []
    pending_tool_calls: dict[str, int] = {}

    for message in messages:
        if isinstance(message, AIMessage):
            thought = _message_text(message)
            for tool_call in message.tool_calls:
                step = AgentStep(
                    thought=thought,
                    action=str(tool_call.get("name") or ""),
                    action_input=tool_call.get("args"),
                )
                steps.append(step)
                pending_tool_calls[str(tool_call.get("id") or "")] = len(steps) - 1
            continue

        if isinstance(message, ToolMessage):
            parsed = _parse_tool_payload(message.content)
            observation = str(parsed.get("observation") or _message_text(message))
            tool_sources = parsed.get("sources") or []

            if isinstance(tool_sources, list):
                sources.extend(item for item in tool_sources if isinstance(item, dict))

            step_index = pending_tool_calls.get(str(message.tool_call_id or ""))
            if step_index is not None:
                old_step = steps[step_index]
                steps[step_index] = AgentStep(
                    thought=old_step.thought,
                    action=old_step.action,
                    action_input=old_step.action_input,
                    observation=observation,
                )

    return steps, _dedupe_sources(sources)


def _last_ai_text(messages: list[Any]) -> str:
    """获取最后一条 AI 回复文本。"""

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return _message_text(message)
    return ""


def _message_text(message: Any) -> str:
    """兼容 LangChain 可能返回的字符串、列表块和消息对象。"""

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _parse_tool_payload(content: Any) -> dict[str, Any]:
    """解析工具返回值；LangChain 有时会把 dict 转成字符串。"""

    if isinstance(content, dict):
        return content

    text = _message_text(content)
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(value, dict):
            return value

    return {"observation": text, "sources": []}


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """去除重复来源片段，避免前端展示一堆重复证据。"""

    seen: set[tuple[str, int]] = set()
    unique: list[dict[str, Any]] = []
    for source in sources:
        key = (str(source.get("source") or ""), int(source.get("chunk") or 0))
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique
