from __future__ import annotations

"""Vision Step：图片分类 + 视觉模型结构化分析。

流程：
1. 图片转 base64
2. 快速分类（轻量 Prompt，~0.3s）
3. 按类型匹配专用 Prompt，调视觉模型输出结构化 JSON
4. JSON 三级清洗

对外暴露：
- classify_and_analyze(image_path, content_type) → dict  JSON 分析结果
- analyze_with_cache(image_path, content_type) → dict  带内容哈希缓存
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)

from src.core.config import settings
from src.media.context import image_data_url_for_vision
from src.media.parse import parse_json
from src.trace import log_step_ctx

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """惰性初始化 OpenAI 客户端（DashScope 兼容接口）。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.vision_api_key,
            base_url=settings.vision_base_url,
        )
    return _client


def _image_data_url(file_path: str, content_type: str) -> str:
    """将图片文件转为 data URL。"""
    return image_data_url_for_vision(file_path, Path(file_path).name)


def _image_content_hash(file_path: str) -> str:
    """计算图片内容的 MD5 哈希，用于缓存。"""
    return hashlib.md5(Path(file_path).read_bytes()).hexdigest()


# ── 公开 API ────────────────────────────────────────────────────


def classify_and_analyze(
    image_path: str,
    content_type: str = "image/png",
) -> dict[str, Any]:
    """Vision Step 完整流程：分类 → 专用分析 → JSON 清洗。

    返回结构化 dict 或降级描述。
    """
    image_url = _image_data_url(image_path, content_type)
    client = _get_client()

    # ① 分类
    t0 = time.perf_counter()
    image_type = _classify(client, image_url)
    classify_ms = (time.perf_counter() - t0) * 1000
    log_step_ctx("vision:classify", ok=True, duration_ms=classify_ms, image_type=image_type)
    logger.info("image_classify: type=%s, duration_ms=%.1f", image_type, classify_ms)

    # ② 匹配专用 Prompt 分析
    prompt = settings.image_analyze_prompts.get(image_type)
    if prompt is None:
        # 未知类型，用界面截图 Prompt 兜底
        prompt = settings.image_analyze_prompts["A"]
        logger.warning("未知图片类型 %s，回退为截图分析", image_type)
        image_type = "A"

    t1 = time.perf_counter()
    raw_text = _analyze(client, image_url, prompt)
    analyze_ms = (time.perf_counter() - t1) * 1000
    log_step_ctx("vision:analyze", ok=True, duration_ms=analyze_ms, raw_len=len(raw_text))
    logger.info("image_analyze: type=%s, duration_ms=%.1f, raw_len=%s", image_type, analyze_ms, len(raw_text))

    # ③ JSON 清洗
    t2 = time.perf_counter()
    result = parse_json(raw_text)
    parse_ms = (time.perf_counter() - t2) * 1000

    parse_level = result.get("_parse_level", 3 if "_parse_error" in result else 2)
    result["_meta"] = {
        "image_type": image_type,
        "classify_ms": round(classify_ms, 1),
        "analyze_ms": round(analyze_ms, 1),
        "parse_ms": round(parse_ms, 1),
        "total_ms": round(classify_ms + analyze_ms + parse_ms, 1),
        "parse_level": parse_level,
    }
    return result


def analyze_with_cache(image_path: str, content_type: str = "image/png") -> dict[str, Any]:
    """带内容哈希缓存的图片分析。相同图片命中缓存直接返回。"""
    file_hash = _image_content_hash(image_path)
    cache_file = settings.analyses_dir / f"{file_hash}.json"

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            cached["_cache_hash"] = file_hash
            logger.info("analysis_cache_hit: hash=%s", file_hash)
            return cached
        except Exception:
            logger.warning("analysis_cache_read_fail: hash=%s", file_hash)

    result = classify_and_analyze(image_path, content_type)
    result["_cache_hash"] = file_hash

    settings.analyses_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("analysis_cache_save: hash=%s", file_hash)
    return result


# ── 内部函数 ─────────────────────────────────────────────────────


def _classify(client: OpenAI, image_url: str) -> str:
    """快速分类图片类型，返回 A/B/C/D/E。"""
    try:
        resp = client.chat.completions.create(
            model=settings.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url={"url": image_url},
                    ),
                    ChatCompletionContentPartTextParam(
                        type="text",
                        text=settings.image_classify_prompt,
                    ),
                ],
            }],
            max_tokens=10,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
        # 提取第一个有效字母
        for ch in raw:
            if ch in "ABCDE":
                return ch
        return "A"
    except Exception:
        log_step_ctx("vision:classify", ok=False, fallback="A")
        logger.exception("图片分类失败，回退为截图分析")
        return "A"


def _analyze(client: OpenAI, image_url: str, prompt: str) -> str:
    """调用视觉模型进行结构化分析，返回原始文本。"""
    resp = client.chat.completions.create(
        model=settings.vision_model,
        messages=[{
            "role": "user",
            "content": [
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={"url": image_url},
                ),
                ChatCompletionContentPartTextParam(
                    type="text",
                    text=prompt,
                ),
            ],
        }],
        max_tokens=2000,
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""
