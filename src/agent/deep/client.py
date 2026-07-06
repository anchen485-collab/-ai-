from __future__ import annotations

"""OpenAI 兼容聊天接口客户端。

DeepSeek 的接口兼容 OpenAI Chat Completions 格式，所以这里保留通用客户端。
后续如果切换到其它兼容服务商，只需要改 base_url、model 和 API Key。
"""

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatResponse:
    """模型返回结果。

    content 用于解析 ReAct 格式；reasoning_content 用于兼容 DeepSeek-R1
    这类推理模型可能返回的额外推理字段。
    """

    content: str
    reasoning_content: str
    raw: dict[str, Any]


class OpenAICompatibleChatClient:
    """最小可用的 OpenAI 兼容 Chat Completions 客户端。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 90,
    ) -> None:
        self.api_key = api_key
        self.base_url = self._normalize_base_url(base_url)
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> ChatResponse:
        """调用 /chat/completions 并返回第一条消息。"""

        if not self.api_key:
            raise RuntimeError("未配置 OpenAI 兼容 API Key。")

        logger.info("调用 OpenAI 兼容模型：model=%s, base_url=%s", self.model, self.base_url)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"模型接口请求失败：HTTP {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"模型接口连接失败：{exc.reason}") from exc

        message = data["choices"][0]["message"]
        content = str(message.get("content") or "").strip()
        logger.info("OpenAI 兼容模型返回成功：content_chars=%s", len(content))
        return ChatResponse(
            content=content,
            reasoning_content=str(message.get("reasoning_content") or "").strip(),
            raw=data,
        )

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """把服务商地址统一整理成包含 /v1 的接口根路径。"""

        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        return url
