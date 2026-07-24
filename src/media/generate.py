from __future__ import annotations

"""DashScope 文生图封装。

DashScope 生成结果中的图片 URL 有有效期，所以这里会先下载到本地，
再把稳定的本地访问路径返回给前端。
"""

import logging
import mimetypes
import base64
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import dashscope
from dashscope.aigc.image_generation import ImageGeneration, Message
from fastapi import HTTPException, status

from src.core.config import settings
from src.storage.oss import download_file, download_json, upload_file, upload_json

logger = logging.getLogger(__name__)

EDIT_REFERENCE_KEYWORDS = (
    "上一张", "上张", "这张", "刚才那张", "刚刚那张", "它", "这个图", "这幅图",
    "原图", "参考图",
)
EDIT_ACTION_KEYWORDS = (
    "修改", "改成", "换成", "替换", "调整", "优化", "重新改",
    "加上", "添加", "去掉", "删除", "移除", "主体不变", "保持",
)


def generate_image(prompt: str, conversation_id: str | None = None) -> dict:
    """根据用户提示词生成图片；必要时自动基于最近生成图做图片编辑。"""

    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图片生成提示词不能为空。",
        )
    cid = _normalize_or_create_conversation_id(conversation_id)

    latest_image = _latest_image_record(cid)
    should_edit = bool(latest_image) and _looks_like_edit_request(clean_prompt)
    if should_edit:
        return _edit_latest_image(clean_prompt, cid, latest_image)

    return _text_to_image(clean_prompt, cid)


def _text_to_image(clean_prompt: str, conversation_id: str) -> dict:
    """普通文生图：没有最近图或用户没有表达修改意图时使用。"""

    if not settings.image_generation_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="未配置 IMAGE_GENERATION_API_KEY 或 DASHSCOPE_API_KEY。",
        )
    if "{WorkspaceId}" in settings.image_generation_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="IMAGE_GENERATION_BASE_URL 仍包含 {WorkspaceId} 占位符，请删除该配置或替换为真实业务空间 ID。",
        )

    if settings.image_generation_base_url:
        # DashScope SDK 使用全局 base_http_api_url；这里只在配置存在时覆盖。
        dashscope.base_http_api_url = settings.image_generation_base_url

    try:
        response = ImageGeneration.call(
            model=settings.image_generation_model,
            api_key=settings.image_generation_api_key,
            # wan2.7-image-pro 官方格式要求 content 是数组，文生图只传 text。
            messages=[Message(role="user", content=[{"text": clean_prompt}])],
            size=settings.image_generation_size,
            n=settings.image_generation_count,
            prompt_extend=settings.image_generation_prompt_extend,
            watermark=settings.image_generation_watermark,
        )
    except Exception as exc:
        logger.exception("图片生成请求失败")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"图片生成请求失败：{exc}",
        ) from exc

    status_code = int(getattr(response, "status_code", 500) or 500)
    if status_code >= 400:
        message = getattr(response, "message", "") or getattr(response, "code", "")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"图片生成失败[{settings.image_generation_model}]：{message or status_code}",
        )

    image_urls = _extract_image_urls(response)
    if not image_urls:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="图片生成成功但未返回图片 URL。",
        )

    saved_images = [_download_generated_image(url) for url in image_urls]
    logger.info(
        "图片生成完成：model=%s count=%s",
        settings.image_generation_model,
        len(saved_images),
    )
    _append_history(
        conversation_id=conversation_id,
        mode="text_to_image",
        model=settings.image_generation_model,
        prompt=clean_prompt,
        images=saved_images,
        source_image_id=None,
    )
    return {
        "model": settings.image_generation_model,
        "mode": "text_to_image",
        "prompt": clean_prompt,
        "conversation_id": conversation_id,
        "source_image_id": None,
        "images": saved_images,
    }


def _edit_latest_image(
    clean_prompt: str,
    conversation_id: str,
    latest_image: dict,
) -> dict:
    """基于当前会话最近生成图进行图片编辑。"""

    if not settings.image_generation_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="未配置 IMAGE_GENERATION_API_KEY 或 DASHSCOPE_API_KEY。",
        )

    endpoint = _image_edit_endpoint()
    source_image_id = str(latest_image.get("image_id") or "")
    source_path = _generated_image_path(source_image_id)
    if not source_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="最近生成的图片文件不存在，无法基于它继续修改。",
        )

    payload = {
        "model": settings.image_edit_model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                "请在参考图基础上修改，尽量保持原图主体、构图和主要风格，"
                                f"只按用户要求调整：{clean_prompt}"
                            )
                        },
                        {"image": _image_file_to_data_url(source_path)},
                    ],
                }
            ]
        },
        "parameters": {
            "size": settings.image_generation_size,
            "n": settings.image_generation_count,
            "prompt_extend": settings.image_edit_prompt_extend,
            "watermark": settings.image_edit_watermark,
        },
    }

    req = urllib.request.Request(
        url=endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.image_generation_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.exception("图片编辑请求失败")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"图片编辑请求失败：{exc}",
        ) from exc

    image_urls = _extract_image_urls(data)
    if not image_urls:
        message = data.get("message") or data.get("code") or data.get("request_id")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"图片编辑未返回图片 URL：{message or '未知错误'}",
        )

    saved_images = [_download_generated_image(url) for url in image_urls]
    _append_history(
        conversation_id=conversation_id,
        mode="image_edit",
        model=settings.image_edit_model,
        prompt=clean_prompt,
        images=saved_images,
        source_image_id=source_image_id,
    )
    logger.info(
        "图片编辑完成：model=%s source=%s count=%s",
        settings.image_edit_model,
        source_image_id,
        len(saved_images),
    )
    return {
        "model": settings.image_edit_model,
        "mode": "image_edit",
        "prompt": clean_prompt,
        "conversation_id": conversation_id,
        "source_image_id": source_image_id,
        "images": saved_images,
    }


def _extract_image_urls(response) -> list[str]:
    """兼容 DashScope 新旧响应结构，提取图片 URL。"""

    urls: list[str] = []
    output = response.get("output", {}) if isinstance(response, dict) else getattr(response, "output", {})

    # wan2.6-t2i 常见结构：output.choices[].message.content[].image
    choices = output.get("choices", []) if hasattr(output, "get") else []
    for choice in choices or []:
        message = choice.get("message", {}) if hasattr(choice, "get") else getattr(choice, "message", {})
        content = message.get("content", []) if hasattr(message, "get") else getattr(message, "content", [])
        if isinstance(content, str):
            continue
        for item in content or []:
            if not hasattr(item, "get"):
                continue
            url = item.get("image") or item.get("url")
            if url:
                urls.append(url)

    # 兼容旧版文生图结构：output.results[].url
    results = output.get("results", []) if hasattr(output, "get") else []
    for item in results or []:
        if hasattr(item, "get") and item.get("url"):
            urls.append(item["url"])

    return urls


def _download_generated_image(image_url: str) -> dict:
    """下载 DashScope 临时图片，保存为本地静态文件。"""

    settings.generated_image_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(image_url, headers={"User-Agent": "qf-agent/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
            content_type = resp.headers.get("Content-Type", "image/png").split(";")[0]
    except Exception as exc:
        logger.exception("下载生成图片失败：%s", image_url)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"下载生成图片失败：{exc}",
        ) from exc

    suffix = _image_suffix(image_url, content_type)
    image_id = f"{uuid4().hex}{suffix}"
    path = settings.generated_image_dir / image_id
    path.write_bytes(content)
    _upload_generated_image_to_oss(path, image_id, content_type)
    return {
        "image_id": image_id,
        "url": f"/api/generated-images/{image_id}",
        "content_type": content_type,
    }


def _normalize_or_create_conversation_id(conversation_id: str | None) -> str:
    """图片生成也使用 UUID 会话 ID，方便自动找到最近生成图。"""

    if not conversation_id:
        return str(uuid4())
    try:
        return str(UUID(conversation_id.strip()))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法 conversation_id。",
        )


def _history_path(conversation_id: str) -> Path:
    settings.generated_image_history_dir.mkdir(parents=True, exist_ok=True)
    return settings.generated_image_history_dir / f"{conversation_id}.json"


def _load_history(conversation_id: str) -> dict:
    path = _history_path(conversation_id)
    if not path.exists():
        history = _load_history_from_oss(conversation_id)
        if history is not None:
            path.write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return history
        return {"conversation_id": conversation_id, "images": [], "latest_image_id": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("conversation_id", conversation_id)
            data.setdefault("images", [])
            data.setdefault("latest_image_id", None)
            return data
    except Exception:
        logger.exception("读取图片生成历史失败：%s", path)
    return {"conversation_id": conversation_id, "images": [], "latest_image_id": None}


def _save_history(conversation_id: str, history: dict) -> None:
    path = _history_path(conversation_id)
    path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _upload_history_to_oss(conversation_id, history)


def _latest_image_record(conversation_id: str) -> dict | None:
    history = _load_history(conversation_id)
    latest_id = history.get("latest_image_id")
    for item in reversed(history.get("images", [])):
        if item.get("image_id") == latest_id:
            return item
    images = history.get("images", [])
    return images[-1] if images else None


def _append_history(
    *,
    conversation_id: str,
    mode: str,
    model: str,
    prompt: str,
    images: list[dict],
    source_image_id: str | None,
) -> None:
    """记录图片生成链路，后续“上一张/这张”可自动引用。"""

    history = _load_history(conversation_id)
    now = datetime.now(timezone.utc).isoformat()
    for image in images:
        image_id = image.get("image_id")
        if not image_id:
            continue
        history["images"].append({
            "image_id": image_id,
            "url": image.get("url"),
            "content_type": image.get("content_type"),
            "prompt": prompt,
            "mode": mode,
            "model": model,
            "source_image_id": source_image_id,
            "created_at": now,
        })
        history["latest_image_id"] = image_id
    _save_history(conversation_id, history)


def _looks_like_edit_request(prompt: str) -> bool:
    """简单规则判断用户是否想修改上一张图。"""

    normalized = prompt.lower()
    return (
        any(keyword.lower() in normalized for keyword in EDIT_REFERENCE_KEYWORDS)
        or any(keyword.lower() in normalized for keyword in EDIT_ACTION_KEYWORDS)
    )


def _generated_image_path(image_id: str) -> Path:
    clean_id = Path(image_id).name
    if clean_id != image_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="图片 id 非法。")
    path = settings.generated_image_dir / clean_id
    if path.exists():
        return path

    try:
        download_file(settings.oss_image_result_prefix, clean_id, path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"从 OSS 读取生成图片失败：{exc}",
        ) from exc
    return path


def ensure_generated_image_local(image_id: str) -> Path:
    """确保生成图片在本地可读；本地缺失时从 OSS 的 image-result 目录恢复。"""

    return _generated_image_path(image_id)


def _upload_generated_image_to_oss(path: Path, image_id: str, content_type: str) -> None:
    """把生成图片同步到 OSS 的 image-result 目录。"""

    try:
        upload_file(settings.oss_image_result_prefix, path, image_id, content_type)
    except Exception as exc:
        logger.exception("生成图片同步 OSS 失败：%s", image_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"生成图片已保存到本地，但同步 OSS 失败：{exc}",
        ) from exc


def _upload_history_to_oss(conversation_id: str, history: dict) -> None:
    """把图片生成历史同步到 OSS 的 history 目录。"""

    try:
        upload_json(settings.oss_history_prefix, f"{conversation_id}.json", history)
    except Exception as exc:
        logger.exception("图片生成历史同步 OSS 失败：%s", conversation_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"图片生成历史同步 OSS 失败：{exc}",
        ) from exc


def _load_history_from_oss(conversation_id: str) -> dict | None:
    """本地历史缺失时，从 OSS 的 history 目录恢复。"""

    try:
        return download_json(settings.oss_history_prefix, f"{conversation_id}.json")
    except Exception:
        logger.exception("从 OSS 读取图片生成历史失败：%s", conversation_id)
        return None


def _image_file_to_data_url(path: Path) -> str:
    content = path.read_bytes()
    content_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _image_edit_endpoint() -> str:
    base_url = settings.image_edit_base_url.strip()
    if not base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="图片编辑需要配置 IMAGE_EDIT_BASE_URL，或配置可用于图片编辑的 IMAGE_GENERATION_BASE_URL。",
        )
    if "{WorkspaceId}" in base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="IMAGE_EDIT_BASE_URL 仍包含 {WorkspaceId} 占位符，请替换为真实业务空间 ID。",
        )
    if base_url.rstrip("/").endswith("/services/aigc/multimodal-generation/generation"):
        return base_url.rstrip("/")
    return base_url.rstrip("/") + "/services/aigc/multimodal-generation/generation"


def _image_suffix(image_url: str, content_type: str) -> str:
    """根据响应类型或 URL 推断图片后缀。"""

    suffix = mimetypes.guess_extension(content_type or "")
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix

    parsed_path = urllib.parse.urlparse(image_url).path
    url_suffix = Path(parsed_path).suffix.lower()
    if url_suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return url_suffix

    return ".png"
