from __future__ import annotations

"""解析模型输出中的 ReAct 指令。"""

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedOutput:
    """模型单轮输出的结构化结果。"""

    thought: str
    action: str | None = None
    action_input: Any = None
    final_answer: str | None = None


def parse_react_output(text: str) -> ParsedOutput:
    """从模型文本中提取 Thought、Action、Action Input 或最终答案。"""

    final_answer = _extract_block(text, "Final Answer")
    if final_answer:
        return ParsedOutput(
            thought=_extract_block(text, "Thought"),
            final_answer=final_answer,
        )

    action = _extract_block(text, "Action")
    action_input_text = _extract_block(text, "Action Input")
    action_input = _parse_action_input(action_input_text)

    return ParsedOutput(
        thought=_extract_block(text, "Thought"),
        action=action,
        action_input=action_input,
    )


def _extract_block(text: str, label: str) -> str:
    """提取某个标签后面的内容，直到下一个 ReAct 标签。"""

    labels = "Thought|Action|Action Input|Observation|Final Answer"
    pattern = rf"{label}\s*:\s*(.*?)(?=\n(?:{labels})\s*:|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_action_input(text: str) -> Any:
    """优先按 JSON 解析 Action Input，失败时保留原始字符串。"""

    if not text:
        return {}

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

