from __future__ import annotations

"""附件模块：上传、读取、文本提取、会话级大文档索引。

对外 API 保持与旧 attachments.py 兼容。
"""

from src.attachments.models import Attachment
from src.attachments.upload import cleanup_all_uploads, cleanup_uploaded_file, save_upload
from src.attachments.loader import ensure_attachment_local, load_attachments, safe_attachment_path
from src.attachments.document import build_attachment_context, cleanup_small_attachments, extract_text


def get_session_store(*args, **kwargs):
    """延迟加载会话索引，避免普通上传图片时提前导入向量库依赖。"""
    from src.attachments.session import get_session_store as _get_session_store

    return _get_session_store(*args, **kwargs)


def get_existing_session_store(*args, **kwargs):
    """延迟读取已有会话索引，不存在时不创建。"""
    from src.attachments.session import get_existing_session_store as _get_existing_session_store

    return _get_existing_session_store(*args, **kwargs)


def set_current_session(*args, **kwargs) -> None:
    """延迟设置当前会话，只有 Agent 推理时才需要会话索引模块。"""
    from src.attachments.session import set_current_session as _set_current_session

    _set_current_session(*args, **kwargs)


def destroy_session(*args, **kwargs) -> None:
    """延迟清理会话索引。"""
    from src.attachments.session import destroy_session as _destroy_session

    _destroy_session(*args, **kwargs)


def cleanup_all_sessions() -> None:
    """延迟清理所有会话索引。"""
    from src.attachments.session import cleanup_all_sessions as _cleanup_all_sessions

    _cleanup_all_sessions()


def search_attachments(*args, **kwargs):
    """延迟加载附件搜索，避免提前导入向量库依赖。"""
    from src.attachments.session import search_attachments as _search_attachments

    return _search_attachments(*args, **kwargs)


def find_attachment_text(*args, **kwargs):
    """延迟加载附件原文查找，适合日期、标题、编号等精确问题。"""
    from src.attachments.session import find_attachment_text as _find_attachment_text

    return _find_attachment_text(*args, **kwargs)


def attachment_response(item: Attachment) -> dict:
    return {
        "attachment_id": item.id,
        "filename": item.filename,
        "content_type": item.content_type,
        "kind": item.kind,
        "url": f"/api/attachments/{item.id}",
    }
