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


def first_env(*names: str, default: str = "") -> str:
    """按优先级读取环境变量，方便兼容不同 API 服务商的命名。"""

    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


class Settings:
    """集中存放各类运行参数；其它模块统一通过 `settings` 单例读取。"""

    kb_source_dir: Path = env_path(
        "KB_SOURCE_DIR", ROOT_DIR / "src" / "rag" / "data"
    )
    chroma_dir: Path = env_path("CHROMA_DIR", ROOT_DIR / "data" / "chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "qf_knowledge_base")
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # 深度思考 Agent 使用 OpenAI 兼容接口。
    # DeepSeek-R1 可配置为：
    # OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com
    # DEEP_AGENT_MODEL=deepseek-reasoner
    openai_compatible_api_key: str = first_env(
        "OPENAI_COMPATIBLE_API_KEY",
        "DEEP_AGENT_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    )
    openai_compatible_base_url: str = os.getenv(
        "OPENAI_COMPATIBLE_BASE_URL", "https://api.deepseek.com"
    )
    deep_agent_model: str = os.getenv("DEEP_AGENT_MODEL", "deepseek-reasoner")
    deep_agent_temperature: float = float(os.getenv("DEEP_AGENT_TEMPERATURE", "0.2"))
    deep_agent_max_steps: int = int(os.getenv("DEEP_AGENT_MAX_STEPS", "8"))
    deep_agent_timeout: int = int(os.getenv("DEEP_AGENT_TIMEOUT", "90"))


# 全局单例，避免在每个模块重复实例化
settings = Settings()
