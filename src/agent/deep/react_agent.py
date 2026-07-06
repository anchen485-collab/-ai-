from __future__ import annotations

"""ReAct 深度思考 Agent 主循环。"""

import logging
from dataclasses import dataclass
from typing import Any

from src.agent.deep.client import OpenAICompatibleChatClient
from src.agent.deep.parser import parse_react_output
from src.agent.deep.prompts import SYSTEM_PROMPT, build_user_prompt
from src.rag.embeddings import SearchHit
from src.tools.rag import get_agent_tools, parse_tool_input, serialize_hits


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentStep:
    """记录一轮 ReAct 执行过程，方便前端和命令行展示。"""

    thought: str
    action: str | None = None
    action_input: Any = None
    observation: str | None = None


@dataclass(frozen=True)
class AgentResult:
    """Agent 最终结果。"""

    answer: str
    steps: list[AgentStep]
    sources: list[dict[str, Any]]


class DeepThinkingAgent:
    """具备 ReAct 和反思机制的深度思考 Agent。"""

    def __init__(
        self,
        client: OpenAICompatibleChatClient,
        max_steps: int = 8,
        temperature: float = 0.2,
    ) -> None:
        self.client = client
        self.max_steps = max_steps
        self.temperature = temperature
        # 工具用 LangChain @tool 定义，初始化 Agent 时集中放进工具列表。
        self.tools = get_agent_tools()

    def run(self, question: str) -> AgentResult:
        """执行 Thought -> Action -> Observation 循环直到最终答案。"""

        logger.info("深度思考 Agent 开始处理问题：%s", question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question)},
        ]
        steps: list[AgentStep] = []
        all_hits: list[SearchHit] = []
        no_progress_count = 0
        last_fingerprint = ""

        for step_index in range(1, self.max_steps + 1):
            logger.info("深度思考 Agent 进入第 %s 步", step_index)
            response = self.client.chat(messages, temperature=self.temperature)
            output = response.content
            parsed = parse_react_output(output)

            messages.append({"role": "assistant", "content": output})

            if parsed.final_answer:
                logger.info("深度思考 Agent 生成最终答案：step=%s", step_index)
                steps.append(AgentStep(thought=parsed.thought))
                return AgentResult(
                    answer=parsed.final_answer,
                    steps=steps,
                    sources=serialize_hits(_dedupe_hits(all_hits)),
                )

            if not parsed.action:
                observation = "输出格式不完整：缺少 Action 或 Final Answer。请按 ReAct 格式继续。"
                logger.warning("深度思考 Agent 输出格式不完整：step=%s", step_index)
                steps.append(AgentStep(thought=parsed.thought, observation=observation))
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                continue

            logger.info(
                "深度思考 Agent 准备执行工具：step=%s, action=%s, input=%s",
                step_index,
                parsed.action,
                parsed.action_input,
            )
            tool_result = self._run_tool(parsed.action, parsed.action_input)
            all_hits.extend(_hits_from_sources(tool_result.get("sources", [])))
            observation = str(tool_result.get("observation") or "")
            logger.info(
                "深度思考 Agent 工具返回：step=%s, hits=%s, observation_chars=%s",
                step_index,
                len(tool_result.get("sources", [])),
                len(observation),
            )

            fingerprint = _fingerprint_sources(tool_result.get("sources", []))
            if not tool_result.get("sources") or fingerprint == last_fingerprint:
                no_progress_count += 1
            else:
                no_progress_count = 0
                last_fingerprint = fingerprint

            if no_progress_count >= 3:
                logger.warning("深度思考 Agent 连续三次检索无进展，触发反思提示")
                observation += (
                    "\n\n反思提示：连续三次检索没有获得新进展。"
                    "请重新审视用户问题，换检索关键词、调整问题范围，"
                    "或者承认证据不足并准备最终回答。"
                )
                no_progress_count = 0

            steps.append(
                AgentStep(
                    thought=parsed.thought,
                    action=parsed.action,
                    action_input=parsed.action_input,
                    observation=observation,
                )
            )
            messages.append({"role": "user", "content": f"Observation: {observation}\n请继续。"})

        return AgentResult(
            answer="已达到最大深度思考轮数，但还没有形成可靠结论。建议缩小问题范围，或补充更多背景信息后再试。",
            steps=steps,
            sources=serialize_hits(_dedupe_hits(all_hits)),
        )

    def _run_tool(self, action: str, action_input: Any) -> dict[str, Any]:
        """从工具列表中找到模型指定的工具并执行。"""

        for agent_tool in self.tools:
            if agent_tool.name != action:
                continue
            return agent_tool.invoke(parse_tool_input(action_input))

        logger.warning("深度思考 Agent 请求了未知工具：%s", action)
        return {
            "observation": f"未知工具：{action}。请只使用可用工具。",
            "sources": [],
        }


def _fingerprint_sources(sources: list[dict[str, Any]]) -> str:
    """用来源和片段号判断工具返回是否有新进展。"""

    return "|".join(f"{item.get('source')}:{item.get('chunk')}" for item in sources[:5])


def _hits_from_sources(sources: list[dict[str, Any]]) -> list[SearchHit]:
    """把工具返回的字典来源还原成 SearchHit，复用原有去重和序列化逻辑。"""

    hits: list[SearchHit] = []
    for item in sources:
        hits.append(
            SearchHit(
                text=str(item.get("text") or ""),
                source=str(item.get("source") or "未知来源"),
                chunk=int(item.get("chunk") or 0),
                distance=(
                    float(item["distance"])
                    if item.get("distance") is not None
                    else None
                ),
            )
        )
    return hits


def _dedupe_hits(hits: list[SearchHit]) -> list[SearchHit]:
    """去除重复来源片段，避免前端展示太多重复证据。"""

    seen: set[tuple[str, int]] = set()
    unique: list[SearchHit] = []
    for hit in hits:
        key = (hit.source, hit.chunk)
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique
