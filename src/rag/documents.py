from __future__ import annotations

# src/rag/documents.py
# 职责：Word(docx) 文档加载 + 切分成文本块（chunks）
# 说明：
# - 切分粒度决定检索召回质量（太粗则噪音多，太细则丢失上下文）
# - chunk_overlap 可以减少边界信息丢失（如跨段句子被切断）
# - 中文场景优先按句号、感叹号等标点切分，避免切断完整句子

"""Word(docx) 知识库加载与切分：把知识库 docx 转成可向量化的 chunks。"""

import hashlib
from pathlib import Path
from typing import List

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import Settings


class DocxIngestor:
    """docx 文档加载与切分器。"""

    def __init__(self, config: Settings) -> None:
        self.__config = config

    def kb_hash(self, source_dir: Path | None = None) -> str:
        """
        计算知识库目录下所有 docx 的联合 MD5 哈希。
        任一文件新增/删除/修改/重命名 → 哈希变化 → 触发新建 collection。
        """
        root = source_dir or self.__config.kb_source_dir
        if not root.exists():
            raise FileNotFoundError(f"知识库目录不存在: {root}")
        h = hashlib.md5()
        # sorted 保证文件遍历顺序稳定，否则同样内容会得到不同哈希
        for path in sorted(root.glob("*.docx")):
            # 文件名纳入哈希，避免重命名场景被识别为未变更
            h.update(path.name.encode("utf-8"))
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        return h.hexdigest()

    def ingest(self, source_dir: Path | None = None) -> List[Document]:
        """
        从知识库目录加载所有 docx → 切分 → 返回 chunks
        :param source_dir: 知识库目录，None 时使用配置中的 kb_source_dir
        :return: 切分后的 Document 列表，可直接用于向量化
        """
        root = source_dir or self.__config.kb_source_dir
        if not root.exists():
            raise FileNotFoundError(f"知识库目录不存在: {root}")

        # sorted 保证构建顺序稳定，进而保证后续 chunk ID 可复现
        documents: List[Document] = []
        for path in sorted(root.glob("*.docx")):
            documents.extend(self.load(str(path)))

        # 切分成 chunks
        return self.split(documents)

    def load(self, docx_path: str) -> List[Document]:
        """
        加载单个 docx 文件
        :param docx_path: docx 文件路径
        :return: Document 列表，每个 Document 包含 page_content 和 metadata（如来源路径）
        """
        docs = Docx2txtLoader(docx_path).load()
        # 标注来源路径，便于后续在回答中溯源
        for doc in docs:
            doc.metadata.setdefault("source", docx_path)
            doc.metadata["source_path"] = docx_path
        return docs

    def split(self, docs: List[Document]) -> List[Document]:
        """
        将每个文档切分成更小的文本块（chunk）
        切分策略：
        - 优先按中文句号、感叹号、问号、分号、逗号等切分
        - 如果 chunk 仍超过 chunk_size，则递归按空格/换行切分
        - 保留 chunk_overlap 个字符的重叠，避免语义断裂
        :param docs: 原始 Document 列表（每个文件至少一个）
        :return: 切分后的 Document 列表（每个 chunk 一个）
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.__config.chunk_size,
            chunk_overlap=self.__config.chunk_overlap,
            is_separator_regex=True,      # 启用正则分隔符
            separators=["(?<=。)", "(?<=！)", "(?<=？)", "(?<=；)", "(?<=，)", " ", "\n"],
        )
        return splitter.split_documents(docs)
