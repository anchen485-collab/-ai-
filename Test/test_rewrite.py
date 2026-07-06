from __future__ import annotations

"""测试查询重写模块：指代词替换 + 问题扩写。

测试层级：
1. has_pronoun — 快速指代词检测（纯函数，无外部依赖）
2. rewrite_query — 查询重写主流程
   - LLM 不可用时的兜底行为
   - LLM 可用时通过 mock 验证重写流程
   - 历史对话消除指代词
   - 边界条件
3. 消息构建 — 验证对话历史正确传入 LLM

LLM 调用通过 urllib.request.urlopen 发出，测试中统一 mock 避免真实网络请求。
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.rag.rewrite import (
    _build_rewrite_messages,
    _call_llm_rewrite,
    has_pronoun,
    rewrite_query,
)


def _fake_response(content: str) -> MagicMock:
    """构造一个支持上下文管理器的假 HTTP 响应，返回指定的 LLM 内容。"""
    body = json.dumps(
        {"choices": [{"message": {"content": content}}]},
        ensure_ascii=False,
    ).encode("utf-8")
    mock = MagicMock()
    mock.read.return_value = body
    mock.__enter__.return_value = mock
    return mock


# =============================================================================
# 1. has_pronoun 指代词检测
# =============================================================================

class TestHasPronoun:
    """纯函数测试，验证常见中文指代词的识别。"""

    @pytest.mark.parametrize(
        "text",
        [
            "他是谁？",
            "她的功能有哪些？",
            "它支持什么格式？",
            "他们怎么注册？",
            "她们可以入驻吗？",
            "它们有什么区别？",
            "其核心能力是什么？",
            "该公司有什么优势？",
            "该平台支持哪些功能？",
            "该项目需要什么资质？",
            "这个功能，它好用吗？",
        ],
    )
    def test_detects_pronouns(self, text: str):
        """包含各种中文指代词的文本应返回 True。"""
        assert has_pronoun(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "全发平台是什么？",
            "怎么注册企业账号？",
            "跨境光伏项目如何找供应商？",
            "发布需求需要什么步骤？",
            "",
            "hello world",
        ],
    )
    def test_no_pronoun_returns_false(self, text: str):
        """无指代词的普通问题应返回 False。"""
        assert has_pronoun(text) is False

    def test_none_input(self):
        """None 输入视为空，返回 False 不抛异常。"""
        assert has_pronoun(None) is False  # type: ignore[arg-type]


# =============================================================================
# 2. rewrite_query 主流程测试
# =============================================================================

class TestRewriteQueryFallback:
    """LLM 不可用时 rewrite_query 的兜底行为。"""

    def test_empty_query_returns_empty(self):
        """空查询应直接返回空 RewriteResult。"""
        result = rewrite_query("")
        assert result.original == ""
        assert result.rewritten == ""
        assert result.changes == []

    def test_no_api_key_returns_original(self, monkeypatch):
        """未配置 API Key 时，返回原问题不变、changes 为空。"""
        # 确保所有可能的 key 环境变量都未设置
        for key in ("OPENAI_COMPATIBLE_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        result = rewrite_query("全发平台是什么？")
        assert result.original == "全发平台是什么？"
        assert result.rewritten == "全发平台是什么？"
        assert result.changes == []

    def test_strips_whitespace(self, monkeypatch):
        """前后空白应被 strip。"""
        for key in ("OPENAI_COMPATIBLE_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        result = rewrite_query("  怎么注册？  ")
        assert result.original == "怎么注册？"


class TestRewriteQueryWithMockLLM:
    """Mock LLM 返回，验证重写主流程的各条路径。"""

    def test_rewrite_replaces_pronouns(self):
        """LLM 重写后的问题应与原文不同，changes 标记为 LLM 重写。"""
        with patch(
            "urllib.request.urlopen",
            return_value=_fake_response("全发平台的注册流程是什么？"),
        ):
            result = rewrite_query("它的注册流程是什么？", api_key="sk-test")

        assert result.original == "它的注册流程是什么？"
        assert result.rewritten == "全发平台的注册流程是什么？"
        assert "LLM 重写" in result.changes

    def test_rewrite_expands_short_query(self):
        """简短问题应被扩写为更完整的表述。"""
        with patch(
            "urllib.request.urlopen",
            return_value=_fake_response("如何在平台进行企业注册？"),
        ):
            result = rewrite_query("怎么注册？", api_key="sk-test")

        assert result.original == "怎么注册？"
        assert result.rewritten == "如何在平台进行企业注册？"
        assert "LLM 重写" in result.changes

    def test_llm_returns_same_text_no_changes(self):
        """LLM 返回原文时，changes 为空，表示无需重写。"""
        with patch(
            "urllib.request.urlopen",
            return_value=_fake_response("全发平台支持哪些频道？"),
        ):
            result = rewrite_query("全发平台支持哪些频道？", api_key="sk-test")

        assert result.rewritten == "全发平台支持哪些频道？"
        assert result.changes == []

    def test_llm_returns_empty_string(self):
        """LLM 返回空字符串时，回退到原文。"""
        with patch(
            "urllib.request.urlopen",
            return_value=_fake_response(""),
        ):
            result = rewrite_query("怎么入驻？", api_key="sk-test")

        assert result.rewritten == "怎么入驻？"
        assert result.changes == []

    def test_llm_returns_whitespace_only(self):
        """LLM 返回纯空白时，回退到原文。"""
        with patch(
            "urllib.request.urlopen",
            return_value=_fake_response("   "),
        ):
            result = rewrite_query("找项目", api_key="sk-test")

        assert result.rewritten == "找项目"
        assert result.changes == []

    def test_llm_call_failure_falls_back(self):
        """LLM 调用抛出异常时，静默回退到原文。"""
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = rewrite_query("它的费用是多少？", api_key="sk-test")

        assert result.rewritten == "它的费用是多少？"
        assert result.changes == []


class TestRewriteQueryWithHistory:
    """验证历史对话能帮助 LLM 消解指代词。"""

    def test_history_passed_to_llm_messages(self):
        """验证历史对话被正确拼接进发送给 LLM 的消息中。"""
        history: list[dict[str, str]] = [
            {"role": "user", "content": "什么是 RAG 模块？"},
            {"role": "assistant", "content": "RAG 模块负责文档检索和向量化存储。"},
        ]

        captured_payload: dict | None = None

        def capture_urlopen(request, timeout=30):
            nonlocal captured_payload
            captured_payload = json.loads(request.data.decode("utf-8"))
            return _fake_response("RAG 模块支持哪些功能？")

        with patch("urllib.request.urlopen", side_effect=capture_urlopen):
            rewrite_query("它支持哪些功能？", history=history, api_key="sk-test")

        assert captured_payload is not None
        messages = captured_payload["messages"]

        # messages 应为: system + history[0] + history[1] + user
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1] == history[0]
        assert messages[2] == history[1]
        assert messages[3]["role"] == "user"
        assert "它支持哪些功能？" in messages[3]["content"]

    def test_history_truncation(self):
        """超长历史应只保留最近 20 条（防止上下文溢出）。"""
        # 构造 30 轮对话 = 60 条消息
        long_history: list[dict[str, str]] = []
        for i in range(30):
            long_history.append({"role": "user", "content": f"第{i}个问题"})
            long_history.append({"role": "assistant", "content": f"第{i}个回答"})

        captured_messages: list[dict[str, str]] = []

        def capture_urlopen(request, timeout=30):
            payload = json.loads(request.data.decode("utf-8"))
            captured_messages.extend(payload["messages"])
            return _fake_response("测试重写结果")

        with patch("urllib.request.urlopen", side_effect=capture_urlopen):
            rewrite_query("那第一个呢？", history=long_history, api_key="sk-test")

        # 1 system + 最多 20 history + 1 user = 最多 22
        assert len(captured_messages) <= 22

    def test_no_history_ok(self):
        """不传 history 也能正常重写。"""
        with patch(
            "urllib.request.urlopen",
            return_value=_fake_response("全发平台怎么发布项目需求？"),
        ):
            result = rewrite_query("怎么发布需求？", api_key="sk-test")

        assert result.rewritten == "全发平台怎么发布项目需求？"


# =============================================================================
# 3. _build_rewrite_messages 单元测试
# =============================================================================

class TestBuildRewriteMessages:
    """验证系统提示、历史、当前问题的组装逻辑。"""

    def test_no_history(self):
        """无历史时只有 system + user 两条消息。"""
        messages = _build_rewrite_messages("怎么注册？", history=None)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "指代词替换" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "怎么注册？" in messages[1]["content"]

    def test_with_history(self):
        """有历史时 system + history + user 顺序拼接。"""
        history: list[dict[str, str]] = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        messages = _build_rewrite_messages("Q2", history=history)

        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Q1"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "A1"
        assert messages[3]["role"] == "user"
        assert "Q2" in messages[3]["content"]

    def test_history_not_mutated(self):
        """不应修改调用者传入的 history 列表。"""
        history: list[dict[str, str]] = [
            {"role": "user", "content": "Q"}
        ]
        original = history.copy()
        _build_rewrite_messages("query", history=history)

        assert history == original
        assert history is not original or history == original


# =============================================================================
# 4. _call_llm_rewrite 边界测试
# =============================================================================

class TestCallLLMRewrite:
    """验证底层 LLM 调用的参数和异常路径。"""

    def test_no_api_key_returns_none(self, monkeypatch):
        """无 API Key 时直接返回 None 不抛异常。"""
        for key in ("OPENAI_COMPATIBLE_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        result = _call_llm_rewrite("test", history=None, api_key="", base_url="", model="")
        assert result is None

    def test_uses_custom_params(self):
        """传入自定义 api_key/base_url/model 应覆盖环境变量。"""
        captured_payload: dict | None = None

        def capture_urlopen(request, timeout=30):
            nonlocal captured_payload
            captured_payload = json.loads(request.data.decode("utf-8"))
            return _fake_response("重写结果")

        with patch("urllib.request.urlopen", side_effect=capture_urlopen):
            _call_llm_rewrite(
                "测试问题",
                history=None,
                api_key="sk-test",
                base_url="https://custom.api.com",
                model="custom-model",
            )

        assert captured_payload is not None
        assert captured_payload["model"] == "custom-model"

    def test_uses_custom_api_key_in_header(self):
        """自定义 api_key 应出现在 Authorization header 中。"""
        captured_request: object | None = None

        def capture_urlopen(request, timeout=30):
            nonlocal captured_request
            captured_request = request
            return _fake_response("结果")

        with patch("urllib.request.urlopen", side_effect=capture_urlopen):
            _call_llm_rewrite(
                "问题",
                history=None,
                api_key="sk-custom-key",
                base_url="https://api.test.com",
                model="gpt-test",
            )

        assert captured_request is not None
        assert captured_request.get_header("Authorization") == "Bearer sk-custom-key"  # type: ignore[union-attr]
