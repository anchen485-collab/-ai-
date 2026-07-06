"""端到端测试：docx 加载 → 文本切分 → 向量化 → 入库。

测试层级：
1. 加载 (DocxIngestor.load)：读取 docx → 返回 Document 列表
2. 切分 (DocxIngestor.split)：Document → chunks
3. 向量化 + 入库 (VectorStoreManager.load_or_build)：chunks → Chroma collection
4. 端到端流水线 (DocxIngestor.ingest + load_or_build)

注意：
- 向量化依赖 DASHSCOPE_API_KEY，未配置时相关测试会被跳过（pytest.skip）。
- chroma_dir 指向 tmp_path，每个测试用例互不影响。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

from src.rag.documents import DocxIngestor
from src.rag.embeddings import VectorStoreManager


# =============================================================================
# 1. 文件加载测试
# =============================================================================

class TestLoad:
    """验证 DocxIngestor.load 能正确读取 docx 并附加 metadata。"""

    def test_load_returns_documents(self, isolated_settings, sample_docx_path: Path):
        """加载 docx 应返回非空 Document 列表。"""
        ingestor = DocxIngestor(isolated_settings)
        docs = ingestor.load(str(sample_docx_path))

        assert isinstance(docs, list)
        assert len(docs) > 0, "docx 加载后应至少返回一个 Document"
        assert all(isinstance(d, Document) for d in docs)

    def test_load_has_page_content(self, isolated_settings, sample_docx_path: Path):
        """每个 Document 的 page_content 应为非空字符串。"""
        ingestor = DocxIngestor(isolated_settings)
        docs = ingestor.load(str(sample_docx_path))

        for doc in docs:
            assert isinstance(doc.page_content, str)
            assert doc.page_content.strip(), "page_content 不应为空字符串"

    def test_load_attaches_source_metadata(self, isolated_settings, sample_docx_path: Path):
        """metadata 应包含 source / source_path，便于后续溯源。"""
        ingestor = DocxIngestor(isolated_settings)
        docs = ingestor.load(str(sample_docx_path))

        for doc in docs:
            assert "source" in doc.metadata
            assert doc.metadata["source_path"] == str(sample_docx_path)

    def test_load_nonexistent_file_raises(self, isolated_settings):
        """加载不存在的文件应抛出异常。"""
        ingestor = DocxIngestor(isolated_settings)
        with pytest.raises(Exception):
            ingestor.load("/no/such/file.docx")


# =============================================================================
# 2. 文本切分测试
# =============================================================================

class TestSplit:
    """验证 DocxIngestor.split 能把长文档切成粒度合理的 chunks。"""

    def test_split_returns_chunks(self, isolated_settings, sample_docx_path: Path):
        """切分后应得到至少一个 chunk。"""
        ingestor = DocxIngestor(isolated_settings)
        docs = ingestor.load(str(sample_docx_path))
        chunks = ingestor.split(docs)

        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_split_respects_chunk_size(self, isolated_settings, sample_docx_path: Path):
        """每个 chunk 的字符数应在 chunk_size 上下浮动（允许 overlap）。"""
        ingestor = DocxIngestor(isolated_settings)
        docs = ingestor.load(str(sample_docx_path))
        chunks = ingestor.split(docs)

        # 单个 chunk 不应远超 chunk_size（容许少量超出，因为按句号切分）
        for chunk in chunks:
            assert len(chunk.page_content) <= isolated_settings.chunk_size * 2

    def test_split_preserves_metadata(self, isolated_settings, sample_docx_path: Path):
        """切分后每个 chunk 仍应携带 source_path，不能丢失来源信息。"""
        ingestor = DocxIngestor(isolated_settings)
        docs = ingestor.load(str(sample_docx_path))
        chunks = ingestor.split(docs)

        for chunk in chunks:
            assert "source_path" in chunk.metadata
            assert chunk.metadata["source_path"] == str(sample_docx_path)


# =============================================================================
# 3. 向量化 + 入库测试（依赖 DASHSCOPE_API_KEY）
# =============================================================================

class TestEmbedAndStore:
    """验证 VectorStoreManager.load_or_build 能完成向量化与入库。"""

    def test_build_new_store(self, isolated_settings, sample_docx_path: Path, dashscope_available):
        """从 chunks 新建向量库：add_documents 时自动向量化，count 应等于 chunks 数。"""
        ingestor = DocxIngestor(isolated_settings)
        docs = ingestor.load(str(sample_docx_path))
        chunks = ingestor.split(docs)

        manager = VectorStoreManager(isolated_settings, DashScopeEmbeddings())
        store = manager.load_or_build("test_hash_001", chunks)

        assert isinstance(store, Chroma)
        assert store._collection.count() == len(chunks), "入库后 collection 数量应等于 chunks 数"

    def test_load_existing_store_without_chunks(
        self, isolated_settings, sample_docx_path: Path, dashscope_available
    ):
        """已存在的向量库再次调用 load_or_build(chunks=None)，应直接复用不报错。"""
        ingestor = DocxIngestor(isolated_settings)
        chunks = ingestor.split(ingestor.load(str(sample_docx_path)))

        manager = VectorStoreManager(isolated_settings, DashScopeEmbeddings())
        # 第一次：新建
        store1 = manager.load_or_build("test_hash_002", chunks)
        count1 = store1._collection.count()

        # 第二次：库已存在，chunks=None
        store2 = manager.load_or_build("test_hash_002", None)
        count2 = store2._collection.count()

        assert count1 == count2 == len(chunks)

    def test_build_without_chunks_raises(self, isolated_settings, dashscope_available):
        """库不存在且未提供 chunks 时应抛出 ValueError。"""
        manager = VectorStoreManager(isolated_settings, DashScopeEmbeddings())
        with pytest.raises(ValueError):
            manager.load_or_build("test_hash_003", None)

    def test_similarity_search_returns_hits(
        self, isolated_settings, sample_docx_path: Path, dashscope_available
    ):
        """入库后对中文 query 做 similarity_search，应返回非空结果。"""
        ingestor = DocxIngestor(isolated_settings)
        chunks = ingestor.split(ingestor.load(str(sample_docx_path)))

        manager = VectorStoreManager(isolated_settings, DashScopeEmbeddings())
        store = manager.load_or_build("test_hash_004", chunks)

        results = store.similarity_search("平台操作", k=3)
        assert len(results) > 0
        assert all(isinstance(d, Document) for d in results)


# =============================================================================
# 4. 端到端流水线测试：ingest() + load_or_build()
# =============================================================================

class TestEndToEndPipeline:
    """组合 DocxIngestor.ingest 与 VectorStoreManager.load_or_build 的完整流水线。"""

    def test_pipeline_load_split_embed_store(
        self, isolated_settings, dashscope_available
    ):
        """完整流水线：扫描目录 → 加载 docx → 切分 → 向量化 → 入库。"""
        # Step 1: 计算知识库哈希（决定 collection 名与持久化目录）
        ingestor = DocxIngestor(isolated_settings)
        file_hash = ingestor.kb_hash()
        assert isinstance(file_hash, str) and len(file_hash) > 0

        # Step 2: 加载 + 切分（合并为 ingest()）
        chunks = ingestor.ingest()
        assert len(chunks) > 0, "知识库切分后应得到非空 chunks"

        # Step 3: 向量化 + 入库
        manager = VectorStoreManager(isolated_settings, DashScopeEmbeddings())
        store = manager.load_or_build(file_hash, chunks)
        assert store._collection.count() == len(chunks)

        # Step 4: 复用同一 hash 第二次调用，应不重复向量化
        store_reused = manager.load_or_build(file_hash, None)
        assert store_reused._collection.count() == len(chunks)

    def test_kb_hash_stable(self, isolated_settings):
        """同一目录多次调用 kb_hash 应返回相同结果。"""
        ingestor = DocxIngestor(isolated_settings)
        assert ingestor.kb_hash() == ingestor.kb_hash()

    def test_kb_hash_changes_on_dir_change(self, isolated_settings, tmp_path: Path):
        """目录内容变化（增删 docx）应导致 kb_hash 变化。"""
        # 构造一份独立的小知识库到 tmp_path
        src_dir = tmp_path / "kb"
        src_dir.mkdir()
        # 复制一个 docx 进去
        sample = isolated_settings.kb_source_dir
        docx_files = sorted(Path(sample).glob("*.docx"))
        if not docx_files:
            pytest.skip("缺少样本 docx")
        (src_dir / docx_files[0].name).write_bytes(docx_files[0].read_bytes())

        ingestor = DocxIngestor(isolated_settings)
        h1 = ingestor.kb_hash(src_dir)

        # 再复制第二个 docx 进去 → 哈希应变化
        if len(docx_files) >= 2:
            (src_dir / docx_files[1].name).write_bytes(docx_files[1].read_bytes())
            h2 = ingestor.kb_hash(src_dir)
            assert h1 != h2, "目录新增 docx 后 kb_hash 必须变化"
