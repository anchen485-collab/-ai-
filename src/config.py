from __future__ import annotations

"""项目配置中心。

所有路径、模型名和检索参数都集中在这里读取，避免散落在业务代码里。
优先读取 .env；没有配置时使用适合当前本机项目的默认值。
"""

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
# 加载项目根目录的 .env，便于本地配置 DASHSCOPE_API_KEY 等敏感信息。
load_dotenv(ROOT_DIR / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    """从环境变量读取路径；未配置时返回默认路径。"""

    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default


class Settings:
    """运行时配置。

    这些属性在模块导入时读取一次。修改 .env 后需要重启服务才会生效。
    """

    # 桌面知识库目录，默认读取 anan 提供的 docx 文件夹。
    kb_source_dir: Path = _path_from_env(
        "KB_SOURCE_DIR", Path(r"C:\Users\Administrator\Desktop\知识库")
    )
    # Chroma 持久化目录。data/ 已加入 .gitignore，不会推到 GitHub。
    chroma_dir: Path = _path_from_env("CHROMA_DIR", ROOT_DIR / "data" / "chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "qf_knowledge_base")
    # 有 DASHSCOPE_API_KEY 时，agent.py 会用该模型生成自然语言回答。
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    # 每次向量检索返回的片段数。
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    # 文档切块参数，影响召回粒度。
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "650"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "90"))


# 全项目共享同一个 settings 实例。
settings = Settings()
