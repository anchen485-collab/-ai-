from __future__ import annotations

"""项目配置：从 .env 文件和环境变量加载运行参数。"""

import os
from pathlib import Path

from dotenv import load_dotenv


# 项目根目录 = 当前文件向上两级 (src/core/config.py -> 项目根)
ROOT_DIR = Path(__file__).resolve().parents[2]
# 启动时加载一次 .env，让本地开发的密钥/路径不进版本库
load_dotenv(ROOT_DIR / ".env")


def env_path(name: str, default: Path) -> Path:
    """读取环境变量中的路径，并展开 ~ 为用户目录。"""
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


class Settings:
    """集中存放各类运行参数；其它模块统一通过 `settings` 单例读取。"""

    kb_source_dir: Path = env_path(
        "KB_SOURCE_DIR", ROOT_DIR / "src" / "rag" / "data"
    )
    chroma_dir: Path = env_path("CHROMA_DIR", ROOT_DIR / "data" / "chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "qf_knowledge_base")
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


# 全局单例，避免在每个模块重复实例化
settings = Settings()
