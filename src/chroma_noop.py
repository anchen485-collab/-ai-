from __future__ import annotations

"""Chroma 遥测空实现。

Chroma 0.5.x 默认会初始化 posthog 遥测。
本项目不需要发送遥测事件，因此用 NoopTelemetry 替换，避免本地日志出现无关 warning。
"""

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoopTelemetry(ProductTelemetryClient):
    """什么都不做的 telemetry client。"""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        # Chroma 会调用这个方法上报事件；这里直接忽略。
        return None
