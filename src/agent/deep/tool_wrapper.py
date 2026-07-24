from __future__ import annotations

"""为深度思考 Agent 的所有工具统一追加反思能力。"""

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from src.agent.deep.reflection import ReflectionManager


def wrap_tools_with_reflection(
    tools: list[Any],
    manager: ReflectionManager,
) -> list[Any]:
    """包装工具列表，让任意工具都能被反思管理器观察。"""

    return [
        _wrap_tool(tool, manager) if isinstance(tool, BaseTool) else tool
        for tool in tools
    ]


def _wrap_tool(tool: BaseTool, manager: ReflectionManager) -> StructuredTool:
    """包装单个 LangChain 工具，保留原工具名称和参数 schema。"""

    def wrapped_tool(**kwargs: Any) -> Any:
        output = tool.invoke(kwargs)
        return manager.observe(
            tool_name=tool.name,
            tool_input=kwargs,
            tool_output=output,
        )

    wrapped_tool.__name__ = f"{tool.name}_with_reflection"
    wrapped_tool.__doc__ = tool.description

    return StructuredTool.from_function(
        func=wrapped_tool,
        name=tool.name,
        description=tool.description or f"{tool.name} tool",
        args_schema=tool.args_schema,
        return_direct=tool.return_direct,
    )
