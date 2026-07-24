from __future__ import annotations

"""会话级附件管理：大文档临时索引 + 检索工具。

流程：
1. 用户上传大文档 → SessionStore.index() 分割向量化入库
2. Agent 推理时调用 search_attachments 工具检索
3. 会话结束时 destroy() 清理临时索引
"""

import contextvars
import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool

from src.core.config import settings
from src.rag.search import SearchHit, serialize_hits

from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

_current_session: contextvars.ContextVar[SessionStore | None] = contextvars.ContextVar(
    "attachment_session", default=None
)
_session_stores: dict[str, SessionStore] = {}


def _session_persist_dir(session_id: str) -> Path:
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
    return settings.chroma_dir / "sessions" / safe_id


def _load_manifest(persist_dir: Path) -> dict | None:
    manifest_path = persist_dir / ".session_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text("utf-8"))
    except Exception:
        return None


class SessionStore:
    """会话级临时向量索引。

    每个 conversation 一个实例，仅索引超过阈值的文档附件。
    会话结束后调用 destroy() 释放资源。
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.collection_name = f"attachments_{session_id}"
        self.persist_dir = _session_persist_dir(session_id)
        self._store: Chroma | None = None
        self._indexed: set[str] = set()
        # 保存分块原文，方便 Agent 按顺序读取大文档靠后的内容。
        self._chunks_by_source: dict[str, list[Document]] = {}
        # 嵌入图片摘要，供 context_hint 列出以便 Agent 精准检索。
        self._image_topics: dict[str, list[str]] = {}
        # 追踪上传文件路径，destroy 时一并清理
        self._attachment_paths: list[Path] = []
        self._attachment_ids: list[str] = []
        # 追踪图片分析缓存哈希，destroy 时一并清理
        self._analysis_cache_hashes: set[str] = set()

    @property
    def _manifest_path(self) -> Path:
        return self.persist_dir / ".session_manifest.json"

    def _persist_manifest(self) -> None:
        manifest = {
            "attachment_paths": [str(p) for p in self._attachment_paths],
            "attachment_ids": self._attachment_ids,
            "analysis_cache_hashes": list(self._analysis_cache_hashes),
        }
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._manifest_path.write_text(json.dumps(manifest), "utf-8")
        except Exception:
            pass

    # ── 索引 ──────────────────────────────────────────────────────

    def index(self, attachment, *, image_descriptions: list[str] | None = None) -> int:
        """对大文档进行分割+向量化入库。返回 chunk 数量，0 表示无需索引或已索引。

        image_descriptions: 文档嵌入图片的 vision 分析描述文本，会作为额外
                           Document 一并入库，使图片内容也可通过检索命中。
        """
        from src.attachments.document import extract_text

        if attachment.kind != "document":
            return 0

        text = extract_text(attachment)
        if len(text) <= settings.attachment_index_threshold_chars:
            return 0

        file_hash = hashlib.md5(attachment.path.read_bytes()).hexdigest()
        if file_hash in self._indexed:
            return 0

        docs = _load_attachment_document(attachment)

        # 嵌入图片描述作为额外 Document 一并入库
        # 必须在 not docs 检查之前添加：当文档加载失败但图片描述存在时，
        # 图片描述仍应被索引，否则大文档的嵌入图片会丢失。
        if image_descriptions:
            topics: list[str] = []
            for desc in image_descriptions:
                docs.append(Document(
                    page_content=desc,
                    metadata={"source": attachment.filename, "kind": "embedded_image"},
                ))
                topic = _extract_image_topic(desc)
                if topic:
                    topics.append(topic)
            if topics:
                self._image_topics[attachment.filename] = topics

        if not docs:
            return 0

        chunks = _split_documents(docs)
        if not chunks:
            return 0

        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks, 1):
            chunk.metadata["source"] = attachment.filename
            chunk.metadata["chunk"] = index
            chunk.metadata["total_chunks"] = total_chunks

        self._chunks_by_source[attachment.filename] = chunks

        store = self._ensure_store()
        start_id = store._collection.count()
        store.add_documents(
            documents=chunks,
            ids=[f"att{start_id + i}" for i in range(len(chunks))],
        )
        self._indexed.add(file_hash)
        self._attachment_paths.append(attachment.path)
        self._attachment_ids.append(attachment.id)
        self._persist_manifest()

        logger.info(
            "session_index: session=%s file=%s chunks=%s images=%s",
            self.session_id, attachment.filename, len(chunks),
            len(image_descriptions) if image_descriptions else 0,
        )
        return len(chunks)

    def context_hint(self) -> str:
        """生成可放进 prompt 的会话附件提示，供后续轮次继续使用。"""

        if not self._chunks_by_source:
            return ""

        lines = [
            "【重要】当前会话已索引以下大型文档，文档内容（含嵌入图片分析描述）已存入向量库：",
        ]
        for idx, filename in enumerate(self._chunks_by_source, 1):
            chunks_list = self._chunks_by_source[filename]
            lines.append(f"{idx}. {filename}（共 {len(chunks_list)} 个片段，含文字和图片描述）")
            image_topics = self._image_topics.get(filename, [])
            if image_topics:
                lines.append(f"   - 嵌入图片共 {len(image_topics)} 张：")
                for i, topic in enumerate(image_topics, 1):
                    lines.append(f"     · 第{i}张：{topic}")
        lines.extend([
            "",
            "用户后续任何关于文档内容的提问（包括图片、图表、流程图、架构图），都必须先调用 search_attachments 检索。",
            "search_attachments 已自动混合文档文字和嵌入图片分析结果，结果中标注 [嵌入图片分析] 的即为图片描述。",
            "如果 search_attachments 未命中，再尝试 read_attachment_chunks 或 find_attachment_text。",
        ])
        return "\n".join(lines)

    def should_index(self, attachment) -> bool:
        """判断附件是否需要索引（文本长度超过阈值）。"""
        from src.attachments.document import extract_text

        if attachment.kind != "document":
            return False
        text = extract_text(attachment)
        return len(text) > settings.attachment_index_threshold_chars

    # ── 检索 ──────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5, kind: str = "") -> list[SearchHit]:
        if self._store is None or self._store._collection.count() == 0:
            return []

        if kind:
            # 明确指定 kind 时只搜该类型，用 Python 侧过滤避免 Chroma filter 兼容问题
            all_results = self._store.similarity_search(query, k=max(k, 8))
            results = [d for d in all_results if d.metadata.get("kind") == kind][:k]
        else:
            # 默认搜索：取足够多结果，在 Python 侧分拆文字/图片再交错合并
            all_results = self._store.similarity_search(query, k=max(k * 3, 15))
            text_results = [d for d in all_results if d.metadata.get("kind") != "embedded_image"]
            img_results = [d for d in all_results if d.metadata.get("kind") == "embedded_image"]

            # 交错合并：每 2 个全文结果后插入 1 个图片结果
            merged: list = []
            img_idx = 0
            for i, doc in enumerate(text_results):
                merged.append(doc)
                if (i + 1) % 2 == 0 and img_idx < len(img_results):
                    merged.append(img_results[img_idx])
                    img_idx += 1
            while img_idx < len(img_results):
                merged.append(img_results[img_idx])
                img_idx += 1
            results = merged[:k]

        hits: list[SearchHit] = []
        for doc in results:
            source = doc.metadata.get("source", "")
            is_image = doc.metadata.get("kind") == "embedded_image"
            text = doc.page_content
            if is_image:
                text = f"[嵌入图片分析] {text}"
            hits.append(SearchHit(
                text=text,
                source=Path(source).name if source else "上传文档",
                chunk=int(doc.metadata.get("chunk", 0) or 0),
            ))
        return hits

    def read_chunks(
        self,
        filename: str = "",
        start: int = 1,
        count: int = 5,
    ) -> dict[str, Any]:
        """按顺序读取大文档片段。

        start 从 1 开始；如果 start <= 0，则从文档末尾倒数读取。
        例如 start=-3,count=3 表示读取最后 3 个片段。
        """

        if not self._chunks_by_source:
            return {
                "observation": "当前会话没有可按顺序读取的大文档。",
                "sources": [],
            }

        source = filename.strip()
        if not source:
            if len(self._chunks_by_source) != 1:
                names = "、".join(self._chunks_by_source.keys())
                return {
                    "observation": f"当前有多个大文档，请指定 filename。可选文件：{names}",
                    "sources": [],
                }
            source = next(iter(self._chunks_by_source))

        chunks = self._chunks_by_source.get(source)
        if chunks is None:
            names = "、".join(self._chunks_by_source.keys())
            return {
                "observation": f"未找到文件：{source}。可选文件：{names}",
                "sources": [],
            }

        total = len(chunks)
        safe_count = max(1, min(int(count or 5), 20))
        if start <= 0:
            start_index = max(0, total + int(start))
        else:
            start_index = max(0, int(start) - 1)
        end_index = min(total, start_index + safe_count)
        selected = chunks[start_index:end_index]

        lines = [
            f"文件：{source}",
            f"总片段数：{total}",
            f"本次读取：片段 {start_index + 1} 到 {end_index}",
        ]
        hits: list[SearchHit] = []
        for chunk in selected:
            chunk_no = int(chunk.metadata.get("chunk", 0) or 0)
            lines.append(f"{chunk_no}. {chunk.page_content[:1200]}")
            hits.append(SearchHit(
                text=chunk.page_content,
                source=source,
                chunk=chunk_no,
            ))

        return {
            "observation": "\n\n".join(lines),
            "sources": serialize_hits(hits),
        }

    def find_text(
        self,
        query: str,
        filename: str = "",
        k: int = 10,
    ) -> dict[str, Any]:
        """按关键词在大文档原文分块中查找，适合日期、标题、编号等精确问题。"""

        if not self._chunks_by_source:
            return {
                "observation": "当前会话没有可查找的大文档。",
                "sources": [],
            }

        source = filename.strip()
        if source:
            source_chunks = self._chunks_by_source.get(source)
            if source_chunks is None:
                names = "、".join(self._chunks_by_source.keys())
                return {
                    "observation": f"未找到文件：{source}。可选文件：{names}",
                    "sources": [],
                }
            candidates = [(source, source_chunks)]
        else:
            candidates = list(self._chunks_by_source.items())

        terms = _expand_search_terms(query)
        safe_k = max(1, min(int(k or 5), 20))
        hits: list[SearchHit] = []
        seen: set[tuple[str, int]] = set()

        for source_name, chunks in candidates:
            for index, chunk in enumerate(chunks):
                text = chunk.page_content
                text_lower = text.lower()
                matched = [term for term in terms if term.lower() in text_lower]
                if not matched:
                    continue

                # 命中标题、日期后，把后续相邻片段也带上，避免只返回标题行。
                end_index = min(len(chunks), index + safe_k)
                for related_chunk in chunks[index:end_index]:
                    chunk_no = int(related_chunk.metadata.get("chunk", 0) or 0)
                    key = (source_name, chunk_no)
                    if key in seen:
                        continue
                    seen.add(key)
                    hits.append(SearchHit(
                        text=related_chunk.page_content,
                        source=source_name,
                        chunk=chunk_no,
                    ))
                    if len(hits) >= safe_k:
                        break
                if len(hits) >= safe_k:
                    break
            if len(hits) >= safe_k:
                break

        if not hits:
            return {
                "observation": (
                    f"未在附件原文中找到这些关键词：{'、'.join(terms)}。"
                    "如果用户问的是语义相近内容，请改用 search_attachments；"
                    "如果用户问的是全文或靠后内容，请改用 read_attachment_chunks。"
                ),
                "sources": [],
            }

        lines = [
            f"原始问题：{query}",
            f"扩展关键词：{'、'.join(terms)}",
            f"命中数量：{len(hits)}",
        ]
        for index, hit in enumerate(hits, 1):
            lines.append(
                f"{index}. 来源：{hit.source} / 片段 {hit.chunk}\n{hit.text[:1200]}"
            )

        return {
            "observation": "\n\n".join(lines),
            "sources": serialize_hits(hits),
        }

    # ── 生命周期 ──────────────────────────────────────────────────

    def _ensure_store(self) -> Chroma:
        from src.rag.engine import build_chroma

        if self._store is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._store = build_chroma(self.collection_name, self.persist_dir)
        return self._store

    def add_analysis_cache_hash(self, file_hash: str) -> None:
        """记录图片分析缓存哈希，destroy 时一并清理。"""
        if file_hash:
            self._analysis_cache_hashes.add(file_hash)
            self._persist_manifest()

    def destroy(self) -> None:
        if self._store:
            try:
                self._store.delete_collection()
            except Exception:
                pass
            self._store = None

        for path in self._attachment_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                logger.warning("session_cleanup_file_fail: %s", path)

        for aid in self._attachment_ids:
            tmp_dir = settings.upload_dir / "docx_images" / aid
            if tmp_dir.exists():
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

        for file_hash in self._analysis_cache_hashes:
            cache_file = settings.analyses_dir / f"{file_hash}.json"
            try:
                cache_file.unlink(missing_ok=True)
            except Exception:
                pass

        if self.persist_dir.exists():
            try:
                shutil.rmtree(self.persist_dir)
            except Exception:
                logger.warning("session_cleanup_fail: %s", self.persist_dir)

        if self.session_id in _session_stores:
            del _session_stores[self.session_id]

        logger.info(
            "session_destroy: session=%s files=%s cache_analyses=%s",
            self.session_id, len(self._attachment_paths), len(self._analysis_cache_hashes),
        )


# ── 全局管理 ──────────────────────────────────────────────────────────


def get_session_store(session_id: str) -> SessionStore:
    if session_id not in _session_stores:
        _session_stores[session_id] = SessionStore(session_id)
    return _session_stores[session_id]


def get_existing_session_store(session_id: str) -> SessionStore | None:
    """返回已存在的会话索引，不存在时不创建新索引。"""

    return _session_stores.get(session_id)


def set_current_session(session: SessionStore | None) -> None:
    _current_session.set(session)


def destroy_session(session_id: str) -> None:
    store = _session_stores.pop(session_id, None)
    if store:
        store.destroy()
        return

    # 服务重启后内存状态丢失，根据 session_id 和 manifest 清理残留磁盘数据
    persist_dir = _session_persist_dir(session_id)
    if not persist_dir.exists():
        return

    manifest = _load_manifest(persist_dir)

    # 清理上传的附件文件
    if manifest:
        for path_str in manifest.get("attachment_paths", []):
            try:
                Path(path_str).unlink(missing_ok=True)
            except Exception:
                pass
        for aid in manifest.get("attachment_ids", []):
            tmp_dir = settings.upload_dir / "docx_images" / aid
            if tmp_dir.exists():
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass
        for file_hash in manifest.get("analysis_cache_hashes", []):
            cache_file = settings.analyses_dir / f"{file_hash}.json"
            try:
                cache_file.unlink(missing_ok=True)
            except Exception:
                pass

    try:
        shutil.rmtree(persist_dir)
    except Exception:
        logger.warning("orphan_session_cleanup_fail: %s", persist_dir)
    else:
        logger.info("orphan_session_cleaned: session=%s manifest=%s",
                    session_id, manifest is not None)


def cleanup_all_sessions() -> None:
    """销毁所有活跃的会话索引。"""
    for session_id in list(_session_stores.keys()):
        destroy_session(session_id)


# ── Agent 工具 ─────────────────────────────────────────────────────────


@tool
def search_attachments(query: str, k: int = 5, kind: str = "") -> dict[str, Any]:
    """检索当前会话上传的文档附件内容（合同、标书等大型文档）。

    当用户问题涉及已上传文件的具体条款、细节、数据、图片、图表时调用此工具。
    小文件内容已直接提供在上下文中，无需通过此工具检索。
    默认搜索已自动混合文档文字和嵌入图片分析结果（标注 [嵌入图片分析]）。

    Args:
        query: 检索关键词，应包含完整的问题或关键词。
        k: 最多返回的片段数量，建议 1 到 10。
        kind: 过滤内容类型。"embedded_image" 仅检索嵌入图片分析描述，
              ""（默认）同时检索文档文字和图片描述。
    """
    session = _current_session.get()
    if session is None or session._store is None:
        return {
            "observation": "当前会话没有已上传的文档附件或附件尚未完成索引。",
            "sources": [],
        }

    hits = session.search(query, k=k, kind=kind)
    if not hits:
        kind_hint = f"（已限定 kind={kind}）" if kind else ""
        return {
            "observation": (
                f"未在附件中找到与 {query} 相关的内容{kind_hint}。"
                "如果用户询问全文、后半部分、靠后的内容或按顺序分析，"
                "请改用 read_attachment_chunks 读取指定片段。"
            ),
            "sources": [],
        }

    kind_tag = f" [kind={kind}]" if kind else ""
    lines = [f"检索词：{query}{kind_tag}", f"命中数量：{len(hits)}"]
    for i, hit in enumerate(hits, 1):
        lines.append(
            f"{i}. 来源：{hit.source} / 片段 {hit.chunk}\n{hit.text[:600]}"
        )

    return {
        "observation": "\n\n".join(lines),
        "sources": serialize_hits(hits),
    }


@tool
def read_attachment_chunks(filename: str = "", start: int = 1, count: int = 5) -> dict[str, Any]:
    """按顺序读取当前会话上传的大文档片段，适合查看靠后内容或总结全文。

    当用户询问"文档后半部分""靠后的内容""第几部分""全文总结/整体分析"
    时，优先使用这个工具补充读取，而不要只依赖 search_attachments。

    Args:
        filename: 文件名。只有一个大文档时可以留空。
        start: 起始片段序号，从 1 开始；传 -3 表示从倒数第 3 个片段开始。
        count: 读取片段数量，建议 1 到 10，最多 20。
    """

    session = _current_session.get()
    if session is None:
        return {
            "observation": "当前会话没有已上传的大文档附件。",
            "sources": [],
        }
    return session.read_chunks(filename=filename, start=start, count=count)


@tool
def find_attachment_text(query: str, filename: str = "", k: int = 10) -> dict[str, Any]:
    """在当前会话上传的大文档原文中查找关键词，适合日期、标题、编号等精确问题。

    当用户问"七月七日做了什么""2026-07-07 的内容""第 3 条"
    这类需要命中文档原文标记的问题时，优先使用这个工具。

    Args:
        query: 用户问题或要查找的关键词。
        filename: 文件名。只有一个大文档时可以留空。
        k: 最多返回的片段数量，建议 1 到 10，最多 20。
    """

    session = _current_session.get()
    if session is None:
        return {
            "observation": "当前会话没有已上传的大文档附件。",
            "sources": [],
        }
    return session.find_text(query=query, filename=filename, k=k)


# ── 文档加载/分割（委托 engine 统一实现）─────────────────────────────


def _load_attachment_document(attachment) -> list[Document]:
    """加载单个附件为 LangChain Document 列表。"""
    from src.rag.engine import load_file

    return load_file(attachment.path, source_label=attachment.filename)


def _split_documents(docs: list[Document]) -> list[Document]:
    from src.rag.engine import build_splitter

    return build_splitter().split_documents(docs)


def _extract_image_topic(desc: str) -> str:
    """从图片描述首行提取主题，如：数据处理流程图、系统架构图等。"""
    m = re.search(r"【用户上传了[^】]*?：(.+?)】", desc)
    return m.group(1).strip() if m else ""


def _expand_search_terms(query: str) -> list[str]:
    """扩展用户查询词，解决"七月七日"与"2026-07-07"写法不一致的问题。"""

    raw = (query or "").strip()
    terms: set[str] = set()
    if raw:
        terms.add(raw)

    for month_text, day_text in re.findall(
        r"([零〇一二两三四五六七八九十]{1,3})月([零〇一二两三四五六七八九十]{1,3})[日号]",
        raw,
    ):
        month = _chinese_number_to_int(month_text)
        day = _chinese_number_to_int(day_text)
        if month and day:
            terms.update(_date_terms(month, day))

    for month_text, day_text in re.findall(r"(\d{1,2})月(\d{1,2})[日号]", raw):
        terms.update(_date_terms(int(month_text), int(day_text)))

    for year, month_text, day_text in re.findall(
        r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?",
        raw,
    ):
        month = int(month_text)
        day = int(day_text)
        terms.update(_date_terms(month, day, year=year))

    return [term for term in terms if term]


def _date_terms(month: int, day: int, year: str | None = None) -> set[str]:
    """生成常见日期写法，方便精确命中文档标题。"""

    terms = {
        f"{month}月{day}日",
        f"{month}月{day}号",
        f"{month:02d}月{day:02d}日",
        f"{month:02d}-{day:02d}",
        f"{month}-{day}",
        f"{month:02d}/{day:02d}",
        f"{month}/{day}",
        f"{month}.{day}",
        f"{month:02d}.{day:02d}",
    }
    if year:
        terms.update({
            f"{year}-{month:02d}-{day:02d}",
            f"{year}/{month:02d}/{day:02d}",
            f"{year}年{month}月{day}日",
        })
    return terms


def _chinese_number_to_int(text: str) -> int | None:
    """把简单中文数字转为整数，覆盖日期里常见的 1-31。"""

    digit_map = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if text in digit_map:
        return digit_map[text]
    if text == "十":
        return 10
    if text.startswith("十"):
        ones = digit_map.get(text[1:], 0) if len(text) > 1 else 0
        return 10 + ones
    if "十" in text:
        tens_text, ones_text = text.split("十", 1)
        tens = digit_map.get(tens_text)
        if tens is None:
            return None
        ones = digit_map.get(ones_text, 0) if ones_text else 0
        return tens * 10 + ones
    return None
