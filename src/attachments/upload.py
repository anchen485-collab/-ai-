from __future__ import annotations

"""统一上传入口：图片委托 media/upload，文本文档本地处理。"""

import logging
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from src.attachments.models import (
    DOC_TYPE,
    DOCX_TYPE,
    TEXT_TYPES,
    Attachment,
    content_type_for_suffix,
    kind_for_suffix,
)
from src.core.config import settings
from src.media.upload import is_image, save_image
from src.storage.oss import delete_file, upload_file

logger = logging.getLogger(__name__)


async def save_upload(file: UploadFile) -> Attachment:
    content_type = (file.content_type or "").lower()
    max_bytes = settings.upload_max_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"附件不能超过 {settings.upload_max_mb}MB。",
        )

    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    if is_image(content_type):
        attachment_id, path = save_image(content, content_type, settings.upload_dir)
        _mirror_upload_to_oss(path, attachment_id, content_type)
        return Attachment(
            id=attachment_id,
            filename=file.filename or attachment_id,
            content_type=content_type,
            kind="image",
            path=path,
        )

    suffix = _suffix_for_upload(file.filename or "", content_type)
    attachment_id = f"{uuid4().hex}{suffix}"
    path = settings.upload_dir / attachment_id
    path.write_bytes(content)
    _mirror_upload_to_oss(path, attachment_id, content_type_for_suffix(suffix))

    return Attachment(
        id=attachment_id,
        filename=file.filename or attachment_id,
        content_type=content_type,
        kind="document",
        path=path,
    )


def _suffix_for_upload(filename: str, content_type: str) -> str:
    if content_type in TEXT_TYPES:
        return TEXT_TYPES[content_type]
    if content_type == DOCX_TYPE:
        return ".docx"
    if content_type == DOC_TYPE:
        return ".doc"

    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".doc", ".docx"}:
        return suffix

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="仅支持 txt、md、doc、docx 文档附件。",
    )


def cleanup_uploaded_file(attachment_id: str) -> bool:
    """删除单个已上传文件。"""
    path = settings.upload_dir / attachment_id
    deleted = False
    try:
        if path.exists():
            path.unlink()
            deleted = True
    except Exception:
        logger.warning("删除上传文件失败: %s", path)

    try:
        deleted = delete_file(settings.oss_uploads_prefix, attachment_id) or deleted
    except Exception:
        logger.warning("删除 OSS 上传文件失败: %s", attachment_id)

    return deleted


def cleanup_all_uploads() -> None:
    """清理所有上传文件及临时目录，在服务关闭时调用。"""
    if not settings.upload_dir.exists():
        return
    try:
        shutil.rmtree(settings.upload_dir)
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info("已清理上传目录: %s", settings.upload_dir)
    except Exception:
        logger.warning("清理上传目录失败: %s", settings.upload_dir)


def _mirror_upload_to_oss(path: Path, attachment_id: str, content_type: str) -> None:
    """把上传附件同步到 OSS 的 uploads 目录。"""

    try:
        upload_file(settings.oss_uploads_prefix, path, attachment_id, content_type)
    except Exception as exc:
        logger.exception("上传附件同步 OSS 失败：%s", attachment_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"附件已保存到本地，但同步 OSS 失败：{exc}",
        ) from exc
