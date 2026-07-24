from __future__ import annotations

"""附件读取与路径校验。"""

from pathlib import Path

from fastapi import HTTPException, status

from src.attachments.models import (
    SUPPORTED_SUFFIXES,
    Attachment,
    content_type_for_suffix,
    kind_for_suffix,
)
from src.core.config import settings
from src.storage.oss import download_file

MB = 1024 * 1024


def load_attachments(attachment_ids: list[str]) -> list[Attachment]:
    if len(attachment_ids) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="单次聊天最多关联 10 个附件。",
        )

    attachments: list[Attachment] = []
    for attachment_id in attachment_ids:
        path = ensure_attachment_local(attachment_id)
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"附件不存在：{attachment_id}。",
            )
        suffix = path.suffix.lower()
        attachments.append(
            Attachment(
                id=attachment_id,
                filename=attachment_id,
                content_type=content_type_for_suffix(suffix),
                kind=kind_for_suffix(suffix),
                path=path,
            )
        )
    validate_attachment_limits(attachments)
    return attachments


def validate_attachment_limits(attachments: list[Attachment]) -> None:
    """限制单次聊天关联的图片数量和附件总大小，避免请求体过大。"""

    image_count = sum(1 for item in attachments if item.kind == "image")
    if image_count > settings.upload_image_max_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"单次聊天最多关联 {settings.upload_image_max_count} 张图片。",
        )

    total_bytes = sum(item.path.stat().st_size for item in attachments)
    max_total_bytes = settings.upload_total_max_mb * MB
    if total_bytes > max_total_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"单次聊天附件总大小不能超过 {settings.upload_total_max_mb}MB。",
        )


def safe_attachment_path(attachment_id: str) -> Path:
    clean_id = Path(attachment_id).name
    if clean_id != attachment_id or Path(clean_id).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法附件 ID。",
        )
    return settings.upload_dir / clean_id


def ensure_attachment_local(attachment_id: str) -> Path:
    """确保附件在本地可读；本地缺失时从 OSS 的 uploads 目录恢复。"""

    path = safe_attachment_path(attachment_id)
    if path.exists():
        return path

    try:
        download_file(settings.oss_uploads_prefix, attachment_id, path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"从 OSS 读取附件失败：{exc}",
        ) from exc
    return path
