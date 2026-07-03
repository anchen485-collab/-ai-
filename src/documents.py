from __future__ import annotations

"""知识库文档解析与切块。

当前知识库来源是桌面“知识库”目录下的 docx 文件。
这里不依赖 python-docx，而是直接读取 docx 内部 XML，减少额外依赖。
"""

import html
import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from .config import settings


@dataclass(frozen=True)
class Chunk:
    """准备写入 Chroma 的文本片段。"""

    id: str
    text: str
    metadata: dict[str, str | int]


def extract_docx_text(path: Path) -> str:
    """从 docx 文件中提取纯文本。

    docx 本质是 zip 包，正文位于 word/document.xml。
    优先按段落读取；如果段落匹配失败，再退化为直接读取所有 w:t 文本节点。
    """

    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")

    paragraphs = re.findall(r"<w:p[\s\S]*?</w:p>", xml)
    parts: list[str] = []
    for paragraph in paragraphs:
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", paragraph)
        line = "".join(html.unescape(item) for item in texts).strip()
        if line:
            parts.append(line)

    if not parts:
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml)
        return "\n".join(html.unescape(item).strip() for item in texts if item.strip())
    return "\n".join(parts)


def load_source_documents(source_dir: Path | None = None) -> list[tuple[Path, str]]:
    """加载知识库目录中的全部 docx 文档。"""

    root = source_dir or settings.kb_source_dir
    if not root.exists():
        raise FileNotFoundError(f"知识库目录不存在: {root}")

    docs: list[tuple[Path, str]] = []
    for path in sorted(root.glob("*.docx")):
        text = extract_docx_text(path)
        if text.strip():
            docs.append((path, text))
    return docs


def split_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """将长文本切成适合向量检索的小块。

    overlap 用来保留相邻块之间的上下文，减少一句话被切断造成的召回损失。
    """

    size = chunk_size or settings.chunk_size
    overlap_size = overlap or settings.chunk_overlap
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[。！？；\n])", text)
        if part and part.strip()
    ]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        # 当前块还能容纳该句子，就继续累积。
        if len(current) + len(sentence) <= size:
            current += sentence
            continue
        # 当前块已满，先保存，再用尾部 overlap 作为下一块前缀。
        if current:
            chunks.append(current.strip())
        prefix = current[-overlap_size:] if overlap_size and current else ""
        current = (prefix + sentence).strip()
        while len(current) > size:
            chunks.append(current[:size].strip())
            current = current[size - overlap_size :].strip() if overlap_size else current[size:].strip()

    if current:
        chunks.append(current.strip())
    return chunks


def build_chunks(source_dir: Path | None = None) -> list[Chunk]:
    """读取源文档并生成带元数据的 Chunk 列表。"""

    chunks: list[Chunk] = []
    for doc_index, (path, text) in enumerate(load_source_documents(source_dir), start=1):
        for chunk_index, chunk_text in enumerate(split_text(text), start=1):
            chunks.append(
                Chunk(
                    id=f"{path.stem}-{doc_index}-{chunk_index}",
                    text=chunk_text,
                    metadata={
                        "source": path.name,
                        "source_path": str(path),
                        "chunk": chunk_index,
                    },
                )
            )
    return chunks
