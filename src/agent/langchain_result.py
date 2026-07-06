from __future__ import annotations

"""整理 LangChain Agent 返回结果的通用工具。"""

import ast
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


@dataclass(frozen=True)
class AgentStep:
    """记录一次工具调用过程，方便前端展示 Agent 执行轨迹。"""

    thought: str
    action: str | None = None
    action_input: Any = None
    observation: str | None = None


@dataclass(frozen=True)
class AgentResult:
    """Agent 的最终结果。"""

    answer: str
    steps: list[AgentStep]
    sources: list[dict[str, Any]]


def extract_steps_and_sources(messages: list[Any]) -> tuple[list[AgentStep], list[dict[str, Any]]]:
    """从 LangChain 消息中提取工具调用步骤和 RAG 来源。"""

    steps: list[AgentStep] = []
    sources: list[dict[str, Any]] = []
    pending_tool_calls: dict[str, int] = {}

    for message in messages:
        if isinstance(message, AIMessage):
            thought = message_text(message)
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
            parsed = parse_tool_payload(message.content)
            observation = str(parsed.get("observation") or message_text(message))
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

    return steps, dedupe_sources(sources)


def last_ai_text(messages: list[Any]) -> str:
    """获取最后一条非工具调用的 AI 回复文本。"""

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message_text(message)
    return ""


def message_text(message: Any) -> str:
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


def parse_tool_payload(content: Any) -> dict[str, Any]:
    """解析工具返回值；LangChain 有时会把 dict 转成字符串。"""

    if isinstance(content, dict):
        return content

    text = message_text(content)
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(value, dict):
            return value

    return {"observation": text, "sources": []}


def dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """去除重复来源片段，避免前端展示重复证据。"""

    seen: set[tuple[str, int]] = set()
    unique: list[dict[str, Any]] = []
    for source in sources:
        key = (str(source.get("source") or ""), int(source.get("chunk") or 0))
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique
