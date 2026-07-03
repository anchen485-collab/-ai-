from __future__ import annotations

"""命令行知识库入库脚本。

运行方式：
    .\.venv311\Scripts\python.exe scripts\ingest.py

它会读取配置中的知识库目录，把 docx 文档解析、切块并写入 Chroma。
"""

import sys
from pathlib import Path

# 直接运行 scripts/ingest.py 时，Python 默认只把 scripts 目录放进 sys.path。
# 这里手动加入项目根目录，保证可以导入 src 包。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.store import ingest


if __name__ == "__main__":
    # reset=True 表示重建 collection，避免旧知识片段残留。
    result = ingest(reset=True)
    print(result)
