from __future__ import annotations

"""JSON → 纯文本上下文，拼接给 Text Agent。

职责：
- 将 Vision Step 产出的结构化 JSON 转换为纯净的文本描述
- 合并用户问题，形成 Text Step 的完整输入
- 图片→base64 data URL 工具函数
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

from src.core.config import settings

MB = 1024 * 1024



def json_to_context(analysis: dict[str, Any]) -> str:
    """将视觉分析 JSON 转换为纯文本上下文。

    不同图片类型输出不同的自然语言模板，Text Agent 据此推理。
    """
    image_type = analysis.get("image_type", "unknown")

    if image_type == "screenshot":
        return _screenshot_context(analysis)
    elif image_type == "document":
        return _document_context(analysis)
    elif image_type == "photo":
        return _photo_context(analysis)
    elif image_type == "chart":
        return _chart_context(analysis)
    elif image_type == "chat_log":
        return _chat_log_context(analysis)
    else:
        # 解析失败兜底：直接用 summary
        return f"[用户上传了一张图片，内容如下]\n{analysis.get('summary', '图片分析结果为空')}"


def has_image_attachments(attachments: list) -> bool:
    """判断附件列表中是否包含图片。"""
    return any(item.kind == "image" for item in attachments)


def image_data_url_for_vision(file_path: str | Path, filename: str = "image") -> str:
    """压缩单张图片，并返回视觉模型可消费的 data URL。"""

    image_bytes = compress_image_for_vision(file_path, filename)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    url = f"data:image/jpeg;base64,{encoded}"
    if len(url.encode("utf-8")) > settings.vision_payload_max_mb * MB:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"图片压缩后仍超过视觉模型请求限制 {settings.vision_payload_max_mb}MB。",
        )
    return url


def compress_image_for_vision(file_path: str | Path, filename: str = "image") -> bytes:
    """把图片压缩成视觉模型更容易消费的 JPEG 字节。"""

    max_side = max(1, settings.vision_image_max_side)
    quality = min(95, max(1, settings.vision_image_jpeg_quality))
    try:
        with Image.open(file_path) as image:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            image = flatten_image(image)
            output = BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            return output.getvalue()
    except (OSError, UnidentifiedImageError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"图片无法解析或压缩：{filename}",
        ) from exc


def flatten_image(image: Image.Image) -> Image.Image:
    """把透明图片铺到白底上，避免转 JPEG 时丢失透明通道导致黑底。"""

    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


# ── 各类型转换模板 ──────────────────────────────────────────────


def _screenshot_context(a: dict) -> str:
    parts = [f"【用户上传了一张界面截图：{a.get('topic', '未知页面')}】"]
    summary = a.get("summary")
    if summary:
        parts.append(f"页面概况：{summary}")

    elements = a.get("elements", [])
    if elements:
        lines = ["界面元素："]
        for el in elements:
            label = el.get("label", "")
            tp = el.get("type", "")
            status = el.get("status", "")
            detail = el.get("detail", "")
            status_tag = f"[{status}]" if status else ""
            detail_text = f"（{detail}）" if detail else ""
            lines.append(f"  - {tp} · {label} {status_tag} {detail_text}".strip())
        parts.append("\n".join(lines))

    errors = a.get("errors", [])
    if errors:
        parts.append("异常信息：" + "；".join(errors))

    ocr = a.get("ocr_text")
    if ocr:
        parts.append(f"页面文字内容：\n{ocr}")

    hint = a.get("user_intent_hint")
    if hint:
        parts.append(f"推测用户意图：{hint}")

    return "\n\n".join(parts)


def _document_context(a: dict) -> str:
    parts = [f"【用户上传了一份文档图片：{a.get('topic', '未知文档')}】"]
    summary = a.get("summary")
    if summary:
        parts.append(f"文档概要：{summary}")

    fields = a.get("fields", [])
    if fields:
        lines = ["提取的字段："]
        for f in fields:
            key = f.get("key", "")
            value = f.get("value", "")
            note = f.get("note", "")
            note_text = f" [{note}]" if note else ""
            lines.append(f"  - {key}：{value}{note_text}")
        parts.append("\n".join(lines))

    tables = a.get("tables", [])
    if tables:
        for t in tables:
            caption = t.get("caption", "")
            headers = t.get("headers", [])
            rows = t.get("rows", [])
            lines = [f"表格{caption + '：' if caption else ''}"]
            if headers:
                lines.append("  | " + " | ".join(headers) + " |")
            for row in rows:
                lines.append("  | " + " | ".join(str(c) for c in row) + " |")
            parts.append("\n".join(lines))

    stamps = a.get("stamps", [])
    if stamps:
        parts.append("印章/签名：" + "、".join(stamps))

    ocr = a.get("ocr_text")
    if ocr:
        parts.append(f"文档完整文字：\n{ocr}")

    hint = a.get("user_intent_hint")
    if hint:
        parts.append(f"推测用户意图：{hint}")

    return "\n\n".join(parts)


def _photo_context(a: dict) -> str:
    parts = [f"【用户上传了一张实物照片：{a.get('topic', '未知')}】"]
    summary = a.get("summary")
    if summary:
        parts.append(f"画面描述：{summary}")

    objects = a.get("objects", [])
    if objects:
        lines = ["画面中的物体/人物："]
        for obj in objects:
            name = obj.get("name", "")
            count = obj.get("count", 1)
            attrs = obj.get("attributes", "")
            pos = obj.get("position", "")
            desc = f"{count}个" if isinstance(count, int) and count > 1 else ""
            lines.append(f"  - {name} {desc}：{attrs}（{pos}）".rstrip("（）"))
        parts.append("\n".join(lines))

    text = a.get("visible_text", [])
    if text:
        parts.append("可见文字/标识：" + "、".join(text))

    anomalies = a.get("anomalies", [])
    if anomalies:
        parts.append("异常细节：" + "；".join(anomalies))

    hint = a.get("user_intent_hint")
    if hint:
        parts.append(f"推测用户意图：{hint}")

    return "\n\n".join(parts)


def _chart_context(a: dict) -> str:
    parts = [f"【用户上传了一张图表：{a.get('topic', '未知图表')}】"]
    chart_type = a.get("chart_type", "")
    if chart_type:
        parts.append(f"图表类型：{chart_type}")

    summary = a.get("summary")
    if summary:
        parts.append(f"图表解读：{summary}")

    insights = a.get("insights", [])
    if insights:
        parts.append("关键洞察：" + "；".join(insights))

    data_points = a.get("data_points", [])
    if data_points:
        lines = ["数据点："]
        for dp in data_points:
            label = dp.get("label", "")
            value = dp.get("value", "")
            trend = dp.get("trend", "")
            trend_icon = {"up": "↑", "down": "↓", "stable": "→"}.get(trend, "")
            lines.append(f"  - {label}：{value} {trend_icon}")
        parts.append("\n".join(lines))

    relationships = a.get("relationships", [])
    if relationships:
        parts.append("节点关系：" + "；".join(relationships))

    ocr = a.get("ocr_text")
    if ocr:
        parts.append(f"图表文字标注：\n{ocr}")

    hint = a.get("user_intent_hint")
    if hint:
        parts.append(f"推测用户意图：{hint}")

    return "\n\n".join(parts)


def _chat_log_context(a: dict) -> str:
    parts = [f"【用户上传了一段聊天记录：{a.get('topic', '未知对话')}】"]
    platform = a.get("platform", "")
    if platform:
        parts.append(f"平台：{platform}")

    participants = a.get("participants", [])
    if participants:
        parts.append("参与方：" + "、".join(participants))

    messages = a.get("messages", [])
    if messages:
        lines = ["对话内容："]
        for msg in messages:
            speaker = msg.get("speaker", "未知")
            content = msg.get("content", "")
            t = msg.get("time", "")
            time_str = f" ({t})" if t else ""
            lines.append(f"  {speaker}{time_str}：{content}")
        parts.append("\n".join(lines))

    conclusions = a.get("key_conclusions", [])
    if conclusions:
        parts.append("关键结论：" + "；".join(conclusions))

    outstanding = a.get("outstanding", [])
    if outstanding:
        parts.append("待办/未决：" + "；".join(outstanding))

    hint = a.get("user_intent_hint")
    if hint:
        parts.append(f"推测用户意图：{hint}")

    return "\n\n".join(parts)
