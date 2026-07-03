from __future__ import annotations

"""Load docx knowledge files and split them into searchable chunks."""

import html
import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from src.core.config import settings


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str | int]


def extract_docx_text(path: Path) -> str:
    """Extract readable text from a docx file without extra parser packages."""

    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")

    paragraphs = re.findall(r"<w:p[\s\S]*?</w:p>", xml)
    lines: list[str] = []
    for paragraph in paragraphs:
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", paragraph)
        line = "".join(html.unescape(item) for item in texts).strip()
        if line:
            lines.append(line)

    if lines:
        return "\n".join(lines)

    texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml)
    return "\n".join(html.unescape(item).strip() for item in texts if item.strip())


def load_documents(source_dir: Path | None = None) -> list[tuple[Path, str]]:
    """Load every docx file from the configured knowledge-base folder."""

    root = source_dir or settings.kb_source_dir
    if not root.exists():
        raise FileNotFoundError(f"知识库目录不存在: {root}")

    docs: list[tuple[Path, str]] = []
    for path in sorted(root.glob("*.docx")):
        text = extract_docx_text(path)
        if text.strip():
            docs.append((path, text))
    return docs


def split_text(text: str) -> list[str]:
    """Split text into overlapping chunks for vector retrieval."""

    size = settings.chunk_size
    overlap = settings.chunk_overlap
    parts = [p.strip() for p in re.split(r"(?<=[。！？；\n])", text) if p.strip()]

    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(current) + len(part) <= size:
            current += part
            continue

        if current:
            chunks.append(current)
        current = (current[-overlap:] + part).strip() if overlap else part

        while len(current) > size:
            chunks.append(current[:size])
            current = current[size - overlap :] if overlap else current[size:]

    if current:
        chunks.append(current)
    return chunks


def build_chunks(source_dir: Path | None = None) -> list[Chunk]:
    """Build Chroma-ready chunks with source metadata."""

    chunks: list[Chunk] = []
    for doc_index, (path, text) in enumerate(load_documents(source_dir), start=1):
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
