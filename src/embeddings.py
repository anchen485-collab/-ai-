from __future__ import annotations

"""本地中文 hashing embedding。

这个 embedding 的目标不是追求最高语义效果，而是保证一期项目：
1. 不需要下载模型。
2. 不需要外部 API Key。
3. 能在 Chroma 中完成稳定的中文静态知识库检索。
"""

import hashlib
import math
import re
from typing import Iterable, List


class HashingChineseEmbedding:
    """Small local embedding function for Chroma smoke-testable retrieval.

    It uses character n-grams and hashing, so it needs no model download or API key.
    This is intentionally simple for the first static-knowledge-base version.
    """

    def __init__(self, dimensions: int = 768) -> None:
        # 维度越高哈希冲突越少，但存储也会更大；768 是比较常见的折中。
        self.dimensions = dimensions

    @staticmethod
    def name() -> str:
        # Chroma 需要 embedding function 有稳定名称，便于持久化配置。
        return "qf_hashing_chinese_embedding"

    @staticmethod
    def build_from_config(config: dict) -> "HashingChineseEmbedding":
        # Chroma 从持久化配置恢复 embedding function 时会调用。
        return HashingChineseEmbedding(dimensions=int(config.get("dimensions", 768)))

    def get_config(self) -> dict:
        # 返回可序列化配置，和 build_from_config 配套使用。
        return {"dimensions": self.dimensions}

    def default_space(self) -> str:
        # 使用 cosine 距离做相似度检索。
        return "cosine"

    def __call__(self, input: Iterable[str]) -> List[List[float]]:
        # Chroma 会批量传入文本列表，因此这里返回向量列表。
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> List[float]:
        """把单条文本转换为归一化向量。"""

        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            # 用 blake2b 把 token 稳定映射到向量下标。
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            # 引入正负号，减少不同 token 全部同向累加带来的偏置。
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        # 归一化后更适合 cosine 检索。
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> list[str]:
        """提取用于 hashing 的 token。

        中文没有空格分词，因此这里同时使用中文单字和 2/3/4-gram。
        """

        normalized = re.sub(r"\s+", "", (text or "").lower())
        tokens: list[str] = []
        tokens.extend(re.findall(r"[a-z0-9_]{2,}", normalized))
        tokens.extend(re.findall(r"[\u4e00-\u9fff]", normalized))
        for size in (2, 3, 4):
            tokens.extend(
                normalized[index : index + size]
                for index in range(max(0, len(normalized) - size + 1))
            )
        return tokens
