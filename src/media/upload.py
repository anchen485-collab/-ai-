from __future__ import annotations

"""图片上传/保存/校验。

从 attachments 模块中抽出的图片专属逻辑，attachments 现在只处理文本文档。
"""

from pathlib import Path
from uuid import uuid4

IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def is_image(content_type: str) -> bool:
    return content_type.lower() in IMAGE_TYPES


def image_suffix(content_type: str) -> str | None:
    return IMAGE_TYPES.get(content_type.lower())


def save_image(content: bytes, content_type: str, upload_dir: Path) -> tuple[str, Path]:
    """保存图片文件，返回 (attachment_id, path)。"""
    suffix = IMAGE_TYPES[content_type.lower()]
    attachment_id = f"{uuid4().hex}{suffix}"
    path = upload_dir / attachment_id
    path.write_bytes(content)
    return attachment_id, path
