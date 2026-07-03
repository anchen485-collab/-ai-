from __future__ import annotations

"""Project settings loaded from .env and environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


class Settings:
    kb_source_dir: Path = env_path(
        "KB_SOURCE_DIR", Path(r"C:\Users\Administrator\Desktop\知识库")
    )
    chroma_dir: Path = env_path("CHROMA_DIR", ROOT_DIR / "data" / "chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "qf_knowledge_base")
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "650"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "90"))


settings = Settings()
