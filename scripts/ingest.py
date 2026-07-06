
from __future__ import annotations

"""命令行知识库入库脚本。

运行方式：
    # 入库（kb_hash 未变则复用，变化则新建 collection 并向量化）
    .\.venv311\Scripts\python.exe scripts\ingest.py

    # 等价别名，保留兼容旧用法
    .\.venv311\Scripts\python.exe scripts\ingest.py --new
"""

import sys
from pathlib import Path

# 直接运行 scripts/ingest.py 时，Python 默认只把 scripts 目录放进 sys.path。
# 这里手动加入项目根目录，保证可以导入 src 包。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.embeddings import ingest, ingest_new


if __name__ == "__main__":
    # 新模型下 hash 不变即复用，全量/增量已合并为同一逻辑
    if "--new" in sys.argv:
        result = ingest_new()
        print("[入库]", result)
    else:
        result = ingest()
        print("[入库]", result)
