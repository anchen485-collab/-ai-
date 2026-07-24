from __future__ import annotations

"""全链路 Trace 机制。

trace_id → route → vision → memory → agent → rag → 耗时
支持按 trace_id 回放分析过程。

设计原则：轻量级，无外部依赖，适合单机服务。
日均万次调用以上时可升级为 OpenTelemetry。

使用 ContextVar 传递 trace_id，任何深层模块无需改函数签名即可上报步骤。
用法：
  set_trace(trace_id)   # 请求入口
  log_step_ctx("step")  # 任意深度模块（自动读取当前 trace_id）
  clear_trace()         # 请求结束
"""

import contextvars
import logging
import threading
import time
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# 内存存储，重启清空。生产环境可换 Redis/DB。
_store: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_max_traces = 10000  # 防止内存泄漏

# 当前请求的 trace_id，通过 call stack 自动传递，无需改函数签名。
_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_id", default=None
)


def new_trace() -> str:
    """生成新的 trace_id 并初始化记录，同时设为当前上下文。"""
    trace_id = uuid4().hex[:16]
    with _lock:
        _store[trace_id] = {
            "trace_id": trace_id,
            "started_at": time.time(),
            "steps": [],
        }
        _trim()
    _current_trace_id.set(trace_id)
    logger.info("trace_start: trace_id=%s", trace_id)
    return trace_id


def set_trace(trace_id: str) -> None:
    """将已有 trace_id 设为当前上下文（用于非入口处接入）。"""
    _current_trace_id.set(trace_id)


def clear_trace() -> None:
    """清除当前 trace 上下文。"""
    _current_trace_id.set(None)


def log_step(
    trace_id: str,
    step: str,
    ok: bool = True,
    duration_ms: float = 0,
    **extra: Any,
) -> None:
    """记录一个步骤到 trace 记录中，同时打结构化日志。"""
    entry = {"step": step, "ok": ok, "duration_ms": round(duration_ms, 1), **extra}
    with _lock:
        if trace_id in _store:
            _store[trace_id]["steps"].append(entry)
    logger.info(
        "trace_step: trace_id=%s step=%s ok=%s duration_ms=%.1f %s",
        trace_id,
        step,
        ok,
        duration_ms,
        " ".join(f"{k}={v}" for k, v in extra.items()),
    )


def log_step_ctx(
    step: str,
    ok: bool = True,
    duration_ms: float = 0,
    **extra: Any,
) -> None:
    """从 ContextVar 读取当前 trace_id 并记录步骤。

    任何深度的模块调用此函数即可上报，无需接收 trace_id 参数。
    """
    trace_id = _current_trace_id.get()
    if trace_id is None:
        return  # 非请求上下文，静默跳过
    log_step(trace_id, step, ok=ok, duration_ms=duration_ms, **extra)


def finish_trace(trace_id: str) -> None:
    """标记 trace 结束，计算总耗时。"""
    with _lock:
        record = _store.get(trace_id)
        if record is None:
            return
        record["total_ms"] = round(
            (time.time() - record["started_at"]) * 1000, 1
        )
    logger.info("trace_end: trace_id=%s total_ms=%s", trace_id, record["total_ms"])
    _current_trace_id.set(None)


def get_trace(trace_id: str) -> dict[str, Any] | None:
    """按 trace_id 查询完整链路记录。"""
    with _lock:
        record = _store.get(trace_id)
    if record is None:
        return None
    total = time.time() - record["started_at"]
    return {
        "trace_id": record["trace_id"],
        "total_ms": round(total * 1000, 1),
        "steps": record["steps"],
    }


def list_traces(limit: int = 50) -> list[dict[str, Any]]:
    """列出最近的 trace，最新的在前。"""
    with _lock:
        ids = list(_store.keys())[-limit:]
    return [
        {"trace_id": tid, "total_ms": round((time.time() - _store[tid]["started_at"]) * 1000, 1)}
        for tid in reversed(ids)
    ]


def _trim() -> None:
    """控制内存用量，超出上限时移除最旧的记录。"""
    while len(_store) > _max_traces:
        oldest = next(iter(_store))
        del _store[oldest]
