from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import settings

logger = logging.getLogger(__name__)


def _file_path(conversation_id: str) -> Path:
    """生成会话记忆文件路径，并确保文件不会逃出记忆目录。"""

    base_dir = settings.memory_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    path = (base_dir / f"{conversation_id}.json").resolve()
    if path.parent != base_dir:
        raise ValueError("非法 conversation_id 路径。")
    return path


def load(conversation_id: str) -> dict | None:
    path = _file_path(conversation_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("读取会话文件失败: %s", path)
        return None


def save(conversation_id: str, data: dict) -> None:
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = _file_path(conversation_id)
    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        logger.exception("写入会话文件失败: %s", path)
