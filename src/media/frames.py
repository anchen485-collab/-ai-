from __future__ import annotations

"""视频关键帧提取。

用于视频多模态分析的预处理步骤：
1. ffmpeg 按时间间隔提取关键帧
2. 逐帧用视觉模型分析
3. 汇总帧描述生成视频内容摘要

当前为占位实现。完整的视频分析需在 analyze.py 中接入帧序列分析流程。
"""

import logging
import subprocess
import tempfile
from pathlib import Path

from src.core.config import settings

logger = logging.getLogger(__name__)


def extract_keyframes(video_path: str, max_frames: int | None = None) -> list[str]:
    """用 ffmpeg 从视频中提取关键帧。

    Args:
        video_path: 视频文件路径
        max_frames: 最大帧数，默认从 settings 读取

    Returns:
        提取的帧图片路径列表（临时目录中）
    """
    max_frames = max_frames or settings.max_video_frames

    output_dir = Path(tempfile.mkdtemp(prefix="video_frames_"))
    output_pattern = str(output_dir / "frame_%03d.jpg")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", f"fps=1",
                "-frames:v", str(max_frames),
                "-q:v", "2",
                output_pattern,
                "-y",
                "-loglevel", "error",
            ],
            check=True,
            timeout=120,
        )
    except subprocess.CalledProcessError:
        logger.exception("ffmpeg 提取关键帧失败：%s", video_path)
    except FileNotFoundError:
        logger.error("未找到 ffmpeg，请安装后再使用视频分析功能")
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg 提取关键帧超时：%s", video_path)

    frames = sorted(output_dir.glob("frame_*.jpg"))
    return [str(f) for f in frames]
