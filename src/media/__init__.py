from __future__ import annotations

"""媒体分析模块：图片上传 → 结构化 JSON → 纯文本上下文。

Vision Step 专注于识图输出结构化 JSON，Text Step 基于 JSON 生成回答。
中间 JSON 可复用（缓存/导出/API），两步可独立优化和测试。
"""
