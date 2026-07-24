from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import HTTPException, status
from src.core.config import settings
from src.memory.store import load, save
from src.trace import log_step_ctx

logger = logging.getLogger(__name__)


def _rounds(messages: list[dict]) -> int:
    """以 user+assistant 为一轮，计算已有轮数。"""
    pairs = 0
    i = 0
    while i < len(messages):
        if (
            messages[i]["role"] == "user"
            and i + 1 < len(messages)
            and messages[i + 1]["role"] == "assistant"
        ):
            pairs += 1
            i += 2
            continue
        i += 1
    return pairs


def normalize_conversation_id(conversation_id: str | None) -> str | None:
    """校验会话 ID，只允许 UUID，避免用户传入路径片段。"""

    if not conversation_id:
        return None

    value = conversation_id.strip()
    if not value:
        return None

    try:
        return str(UUID(value))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法 conversation_id。",
        )


def _summarize(conversation: dict, new_rounds: int) -> str | None:
    """调用 LLM 生成/更新摘要并写回，返回新摘要文本。"""
    from langchain_community.chat_models import ChatTongyi

    if not settings.dashscope_api_key:
        logger.warning("缺少 DASHSCOPE_API_KEY，跳过摘要生成")
        return conversation.get("summary")

    messages = conversation["messages"]
    existing = conversation.get("summary")
    # 只摘要本次新增的轮数对应的消息
    start = conversation.get("summary_rounds", 0) * 2
    new_messages = messages[start:]

    lines = []
    if existing:
        lines.append(f"已有摘要：{existing}")
        lines.append("")
    lines.append("以下是新的对话内容：")
    for m in new_messages:
        role = "用户" if m["role"] == "user" else "助手"
        lines.append(f"{role}：{m['content']}")

    prompt = (
        "你是一个摘要助手。请把对话内容压缩为一段简洁的摘要（不超过200字），"
        "保留关键事实：用户身份、目标、偏好、已确认信息、已上传的文档附件及其中讨论的图片/流程/数据。\n\n"
        + "\n".join(lines)
    )

    try:
        model = ChatTongyi(model=settings.llm_model, api_key=settings.dashscope_api_key)
        result = model.invoke(prompt)
        summary = result.content.strip() if hasattr(result, "content") else str(result).strip()
    except Exception:
        logger.exception("摘要生成失败")
        return existing

    conversation["summary"] = summary
    conversation["summary_rounds"] = conversation.get("summary_rounds", 0) + new_rounds
    return summary


def build_context(conversation_id: str | None) -> tuple[str, list[dict]]:
    """返回 (conversation_id, context_messages)。

    如果 conversation_id 为 None 或不存在，生成新 ID 并返回空上下文。
    context_messages 包含：摘要 system 消息（如有）+ 最近 N 轮完整对话。
    """
    should_summarize = False
    conversation_id = normalize_conversation_id(conversation_id)

    if not conversation_id:
        conversation_id = str(uuid4())
        log_step_ctx("memory:load", ok=True, duration_ms=0, rounds=0, summary=False)
        return conversation_id, []

    conversation = load(conversation_id)
    if conversation is None:
        log_step_ctx("memory:load", ok=True, duration_ms=0, rounds=0, summary=False)
        return conversation_id, []

    total = _rounds(conversation["messages"])
    new_rounds = total - conversation.get("summary_rounds", 0)
    if new_rounds >= settings.memory_summary_rounds:
        should_summarize = True

    if should_summarize:
        new_rounds = total - conversation.get("summary_rounds", 0)
        _summarize(conversation, new_rounds)
        save(conversation_id, conversation)

    context: list[dict] = []
    if conversation.get("summary"):
        context.append({"role": "system", "content": f"[历史对话摘要] {conversation['summary']}"})

    if conversation.get("has_attachments"):
        has_large = conversation.get("has_large_attachments", False)
        if has_large:
            context.append({
                "role": "system",
                "content": (
                    "当前会话中用户已上传文档附件（含嵌入图片）。"
                    "小文档的完整内容（文字和图片描述）已在上面的对话历史中直接展示，先检查历史对话；"
                    "大文档仅展示了开头预览，完整文字和嵌入图片分析描述已向量化索引，"
                    "需调用 search_attachments 检索细节。未命中时再尝试 read_attachment_chunks 或 find_attachment_text。"
                    "不要凭记忆或猜测回答。"
                ),
            })
        else:
            context.append({
                "role": "system",
                "content": (
                    "当前会话中用户已上传文档附件（含嵌入图片）。"
                    "文档完整内容（文字和图片描述）已在上面的对话历史中直接展示，"
                    "请直接在历史对话中查找相关信息回答用户问题，不要凭记忆或猜测回答。"
                ),
            })

    # 取最近 N 轮完整消息
    keep_rounds = settings.memory_recent_rounds
    keep_messages = settings.memory_recent_rounds * 2
    recent = conversation["messages"][-keep_messages:]
    context.extend(recent)

    log_step_ctx("memory:load", ok=True, rounds=total,
                 has_summary=bool(conversation.get("summary")),
                 context_msgs=len(context))
    return conversation_id, context


def save_turn(conversation_id: str, question: str, answer: str, attachment_context: str = "") -> None:
    """保存本轮对话，追加到会话文件。"""
    safe_conversation_id = normalize_conversation_id(conversation_id)
    if safe_conversation_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法 conversation_id。",
        )
    conversation_id = safe_conversation_id
    conversation = load(conversation_id)
    if conversation is None:
        now = datetime.now(timezone.utc).isoformat()
        conversation = {
            "conversation_id": conversation_id,
            "messages": [],
            "summary": None,
            "summary_rounds": 0,
            "created_at": now,
            "updated_at": now,
        }
    t0 = time.perf_counter()
    user_content = f"{question}\n\n{attachment_context}" if attachment_context else question
    conversation["messages"].append({"role": "user", "content": user_content})
    conversation["messages"].append({"role": "assistant", "content": answer})
    if attachment_context:
        conversation["has_attachments"] = True
        # 大文档走 Chroma 索引（上下文中含"search_attachments"引导词），小文档完整内容在对话历史中
        if "search_attachments" in attachment_context:
            conversation["has_large_attachments"] = True
    save(conversation_id, conversation)
    log_step_ctx("memory:save", ok=True, duration_ms=(time.perf_counter() - t0) * 1000)
