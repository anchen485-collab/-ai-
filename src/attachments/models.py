from __future__ import annotations

"""附件数据模型与类型常量。"""

from dataclasses import dataclass
from pathlib import Path

from src.media.upload import IMAGE_SUFFIXES, IMAGE_TYPES

TEXT_TYPES = {
    "text/plain": ".txt",
    "text/markdown": ".md",
}
DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DOCX_SUFFIX = ".docx"
DOC_TYPE = "application/msword"
DOC_SUFFIX = ".doc"
SUPPORTED_SUFFIXES = {".txt", ".md", ".doc", ".docx"} | IMAGE_SUFFIXES


@dataclass(frozen=True)
class Attachment:
    """聊天接口可使用的附件元数据。"""

    id: str
    filename: str
    content_type: str
    kind: str
    path: Path


def kind_for_suffix(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in {".txt", ".md", ".doc", ".docx"}:
        return "document"
    return "unknown"


def content_type_for_suffix(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".doc": DOC_TYPE,
        ".docx": DOCX_TYPE,
    }.get(suffix.lower(), "application/octet-stream")
