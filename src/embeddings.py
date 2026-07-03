from __future__ import annotations

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
        self.dimensions = dimensions

    @staticmethod
    def name() -> str:
        return "qf_hashing_chinese_embedding"

    @staticmethod
    def build_from_config(config: dict) -> "HashingChineseEmbedding":
        return HashingChineseEmbedding(dimensions=int(config.get("dimensions", 768)))

    def get_config(self) -> dict:
        return {"dimensions": self.dimensions}

    def default_space(self) -> str:
        return "cosine"

    def __call__(self, input: Iterable[str]) -> List[List[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> list[str]:
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
