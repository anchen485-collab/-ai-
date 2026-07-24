from __future__ import annotations

"""深度思考 Agent 的通用反思管理器。"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


NO_RESULT_PATTERNS = (
    "没有检索到",
    "未找到",
    "无结果",
    "没有找到",
    "知识库中未",
    "缺少",
    "失败",
    "error",
    "not found",
    "no result",
    "no data",
)


@dataclass
class ReflectionManager:
    """统一记录所有工具调用的进展情况。"""

    input_fingerprints: list[str] = field(default_factory=list)
    output_fingerprints: list[str] = field(default_factory=list)
    no_progress_count: int = 0
    threshold: int = 3

    def reset(self) -> None:
        """每次处理新问题前重置状态，避免不同请求互相影响。"""

        self.input_fingerprints.clear()
        self.output_fingerprints.clear()
        self.no_progress_count = 0

    def observe(self, tool_name: str, tool_input: Any, tool_output: Any) -> Any:
        """观察一次工具调用结果，必要时给工具输出追加反思提示。"""

        input_fingerprint = _fingerprint({"tool": tool_name, "input": tool_input})
        output_fingerprint = _output_fingerprint(tool_output)
        reasons = self._no_progress_reasons(
            input_fingerprint=input_fingerprint,
            output_fingerprint=output_fingerprint,
            tool_output=tool_output,
        )

        if reasons:
            self.no_progress_count += 1
            logger.info(
                "深度 Agent 工具调用无新进展：tool=%s, count=%s, reasons=%s",
                tool_name,
                self.no_progress_count,
                "；".join(reasons),
            )
        else:
            self.no_progress_count = 0

        self._remember(input_fingerprint, output_fingerprint)
        if self.no_progress_count < self.threshold:
            return tool_output

        self.no_progress_count = 0
        logger.warning("深度 Agent 连续工具调用无进展，触发反思提示：tool=%s", tool_name)
        return _append_reflection(tool_output, _reflection_prompt(tool_name, reasons))

    def _no_progress_reasons(
        self,
        input_fingerprint: str,
        output_fingerprint: str,
        tool_output: Any,
    ) -> list[str]:
        """判断本次工具调用没有新进展的原因。"""

        reasons: list[str] = []
        if not _has_result(tool_output):
            reasons.append("工具没有返回有效结果")
        if input_fingerprint in self.input_fingerprints:
            reasons.append("工具入参与之前重复")
        if output_fingerprint and output_fingerprint in self.output_fingerprints:
            reasons.append("工具返回内容与之前重复")
        return reasons

    def _remember(self, input_fingerprint: str, output_fingerprint: str) -> None:
        """只保留最近的调用指纹，避免状态无限增长。"""

        if input_fingerprint:
            self.input_fingerprints.append(input_fingerprint)
        if output_fingerprint:
            self.output_fingerprints.append(output_fingerprint)
        self.input_fingerprints[:] = self.input_fingerprints[-20:]
        self.output_fingerprints[:] = self.output_fingerprints[-20:]


def _has_result(output: Any) -> bool:
    """用通用规则判断工具是否返回了有效信息。"""

    if output is None:
        return False
    if isinstance(output, dict):
        if "sources" in output:
            return bool(output.get("sources"))
        if "results" in output:
            return bool(output.get("results"))
        if "items" in output:
            return bool(output.get("items"))
        if "data" in output and isinstance(output.get("data"), (list, dict, str)):
            return bool(output.get("data"))
        text = _text_for_detection(output)
        return bool(text) and not _looks_like_no_result(text)
    if isinstance(output, (list, tuple, set)):
        return bool(output)
    if isinstance(output, str):
        text = output.strip()
        return bool(text) and not _looks_like_no_result(text)
    return True


def _output_fingerprint(output: Any) -> str:
    """生成工具输出指纹，用于判断重复结果。"""

    if isinstance(output, dict) and isinstance(output.get("sources"), list):
        source_parts = [
            f"{item.get('source')}:{item.get('chunk')}"
            for item in output["sources"][:5]
            if isinstance(item, dict)
        ]
        if source_parts:
            return "|".join(source_parts)
    return _fingerprint(output)


def _append_reflection(output: Any, prompt: str) -> Any:
    """把反思提示追加到不同形态的工具输出中。"""

    if isinstance(output, dict):
        updated = dict(output)
        observation = str(updated.get("observation") or _text_for_detection(output))
        updated["observation"] = f"{observation}\n\n{prompt}".strip()
        return updated
    if isinstance(output, str):
        return f"{output}\n\n{prompt}".strip()
    return {
        "observation": f"{_text_for_detection(output)}\n\n{prompt}".strip(),
        "raw": output,
    }


def _reflection_prompt(tool_name: str, reasons: list[str]) -> str:
    """生成给模型看的通用反思提示。"""

    reason_text = "；".join(reasons) if reasons else "连续工具调用没有获得新信息"
    return (
        "[REFLECTION_PROMPT]\n"
        "反思提示：你已经连续 3 次工具调用没有获得新进展。\n"
        f"最近工具：{tool_name}。\n"
        f"无进展原因：{reason_text}。\n"
        "请不要继续重复相同工具入参或相同策略。请重新审视用户问题，并从下面策略中选择一种：\n"
        "1. 换一组更具体或更宽泛的参数。\n"
        "2. 改用另一个更合适的工具。\n"
        "3. 把问题拆成子问题后分别调用工具。\n"
        "4. 如果现有工具确实没有依据，请停止重复调用，并明确说明证据不足。"
    )


def _fingerprint(value: Any) -> str:
    """把任意值压缩成稳定文本指纹。"""

    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    return re.sub(r"\s+", "", text.lower())[:1500]


def _text_for_detection(value: Any) -> str:
    """把任意工具输出转成用于判断的文本。"""

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _looks_like_no_result(text: str) -> bool:
    """判断文本是否明显表示无结果或失败。"""

    normalized = text.lower()
    return any(pattern in normalized for pattern in NO_RESULT_PATTERNS)
