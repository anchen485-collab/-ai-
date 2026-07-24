from __future__ import annotations

"""阿里云 OSS 存储封装。

当前只负责 agent 相关静态资源：
uploads、image-result、history。
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.core.config import settings

logger = logging.getLogger(__name__)


def is_oss_enabled() -> bool:
    """判断是否启用 OSS。未配置时业务继续使用本地文件。"""

    return settings.oss_enabled


def upload_file(prefix: str, local_path: Path, object_name: str, content_type: str = "") -> str | None:
    """上传本地文件到 OSS，返回 object key；未启用 OSS 时返回 None。"""

    if not is_oss_enabled():
        return None

    key = object_key(prefix, object_name)
    bucket = _bucket()
    headers = {"Content-Type": content_type} if content_type else None
    bucket.put_object_from_file(key, str(local_path), headers=headers)
    logger.info("OSS 上传完成：%s -> %s", local_path, key)
    return key


def upload_json(prefix: str, object_name: str, data: dict[str, Any]) -> str | None:
    """把 JSON 数据写入 OSS。"""

    if not is_oss_enabled():
        return None

    key = object_key(prefix, object_name)
    content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    _bucket().put_object(key, content, headers={"Content-Type": "application/json; charset=utf-8"})
    logger.info("OSS JSON 上传完成：%s", key)
    return key


def download_file(prefix: str, object_name: str, local_path: Path) -> bool:
    """从 OSS 下载文件到本地；对象不存在时返回 False。"""

    if not is_oss_enabled():
        return False

    key = object_key(prefix, object_name)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _bucket().get_object_to_file(key, str(local_path))
        logger.info("OSS 下载完成：%s -> %s", key, local_path)
        return True
    except Exception as exc:
        if _is_no_such_key(exc):
            return False
        raise


def download_json(prefix: str, object_name: str) -> dict[str, Any] | None:
    """从 OSS 读取 JSON；对象不存在或内容非法时返回 None。"""

    if not is_oss_enabled():
        return None

    key = object_key(prefix, object_name)
    try:
        content = _bucket().get_object(key).read()
    except Exception as exc:
        if _is_no_such_key(exc):
            return None
        raise

    try:
        data = json.loads(content.decode("utf-8"))
    except Exception:
        logger.warning("OSS JSON 解析失败：%s", key)
        return None
    return data if isinstance(data, dict) else None


def delete_file(prefix: str, object_name: str) -> bool:
    """删除 OSS 文件。未启用 OSS 时返回 False。"""

    if not is_oss_enabled():
        return False

    key = object_key(prefix, object_name)
    try:
        _bucket().delete_object(key)
        logger.info("OSS 删除完成：%s", key)
        return True
    except Exception as exc:
        if _is_no_such_key(exc):
            return False
        raise


def object_key(prefix: str, object_name: str) -> str:
    """生成 OSS object key，防止外部传入路径穿越。"""

    clean_prefix = prefix.strip("/")
    clean_name = Path(object_name).name
    if clean_name != object_name:
        raise ValueError("非法 OSS 对象名。")
    return f"{clean_prefix}/{clean_name}" if clean_prefix else clean_name


def _bucket():
    """创建 OSS bucket 客户端。"""

    _validate_config()
    try:
        import oss2
    except ImportError as exc:
        raise RuntimeError("启用 OSS 需要安装 oss2，请先执行 pip install -r requirements.txt。") from exc

    auth = oss2.Auth(settings.oss_access_key_id, settings.oss_access_key_secret)
    return oss2.Bucket(auth, settings.oss_endpoint, settings.oss_bucket)


def _validate_config() -> None:
    missing = [
        name
        for name, value in {
            "OSS_ENDPOINT": settings.oss_endpoint,
            "OSS_BUCKET": settings.oss_bucket,
            "OSS_ACCESS_KEY_ID": settings.oss_access_key_id,
            "OSS_ACCESS_KEY_SECRET": settings.oss_access_key_secret,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"OSS 已启用，但缺少配置：{', '.join(missing)}。")


def _is_no_such_key(exc: Exception) -> bool:
    """兼容 oss2 的对象不存在异常判断。"""

    return exc.__class__.__name__ == "NoSuchKey"
