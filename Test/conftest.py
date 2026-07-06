"""pytest 全局夹具与 sys.path 配置。

直接运行 pytest 时，项目根目录不一定在 sys.path 中。
这里把项目根加入 sys.path，使 `from src.* import ...` 在测试中可用。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 项目根目录 = Test 目录上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """返回项目根目录。"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def sample_docx_path(project_root: Path) -> Path:
    """返回一个真实 docx 样本文件路径，用于加载/切分测试。"""
    candidate = project_root / "src" / "rag" / "data" / "全球发展线上平台操作指引知识库.docx"
    if not candidate.exists():
        pytest.skip(f"测试样本 docx 不存在: {candidate}")
    return candidate


@pytest.fixture(scope="session")
def kb_source_dir(project_root: Path) -> Path:
    """返回知识库源目录（含多个 docx），用于入库集成测试。"""
    candidate = project_root / "src" / "rag" / "data"
    if not candidate.exists() or not any(candidate.glob("*.docx")):
        pytest.skip(f"知识库目录为空或不存在: {candidate}")
    return candidate


@pytest.fixture()
def isolated_settings(tmp_path: Path, kb_source_dir: Path):
    """构造一份隔离的 Settings 实例：
    - kb_source_dir 指向真实数据目录
    - chroma_dir 指向 tmp_path，避免污染真实向量库
    每次 test 都拿到一个新的临时 chroma 目录。
    """
    from src.core.config import Settings

    s = Settings()
    s.kb_source_dir = kb_source_dir
    s.chroma_dir = tmp_path / "chroma"
    s.chunk_size = 500
    s.chunk_overlap = 50
    return s


@pytest.fixture()
def dashscope_available() -> bool:
    """检查 DASHSCOPE_API_KEY 是否可用；缺 key 时跳过向量化相关测试。"""
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY 未配置，跳过依赖在线 embedding 的测试")
    return True
