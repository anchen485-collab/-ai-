from __future__ import annotations

"""知识库文档加载与切分：扫描 kb_source_dir 下所有文档（txt/md/docx）→ chunks。

加载、分割均委托 rag/engine.py 统一实现。
"""

import hashlib
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.core.config import Settings
from src.rag.engine import build_splitter, load_files

logger = logging.getLogger(__name__)


class DocxIngestor:
    """知识库文档加载与切分器。"""

    def __init__(self, config: Settings) -> None:
        self.__config = config

    def kb_hash(self, source_dir: Path | None = None) -> str:
        """计算知识库目录下所有文档的联合 MD5 哈希。"""
        root = source_dir or self.__config.kb_source_dir
        if not root.exists():
            raise FileNotFoundError(f"知识库目录不存在: {root}")
        h = hashlib.md5()
        for path in self._kb_paths(root):
            h.update(path.name.encode("utf-8"))
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        return h.hexdigest()

    def ingest(self, source_dir: Path | None = None) -> List[Document]:
        """从知识库目录加载所有文档 → 切分 → 返回 chunks。"""
        root = source_dir or self.__config.kb_source_dir
        if not root.exists():
            raise FileNotFoundError(f"知识库目录不存在: {root}")

        docs = load_files(self._kb_paths(root))
        return build_splitter().split_documents(docs) if docs else []

    def _kb_paths(self, root: Path) -> list[Path]:
        """返回知识库目录下所有支持的文档（txt/md/docx），跳过临时文件。"""
        patterns = ["*.txt", "*.md", "*.docx"]
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(root.glob(pattern))
        return sorted(p for p in paths if not p.name.startswith("~$"))
