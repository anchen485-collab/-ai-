from __future__ import annotations

"""A tiny local Chinese-friendly embedding function for Chroma."""

import hashlib
import math
import re
from typing import Iterable


class HashingChineseEmbedding:
    """Hash Chinese characters and n-grams into a normalized vector.

    This keeps the first version simple: no model download, no GPU, no API key.
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

    def __call__(self, input: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in input]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions

        for token in self.tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        return vector if norm == 0 else [value / norm for value in vector]

    def tokens(self, text: str) -> list[str]:
        text = re.sub(r"\s+", "", (text or "").lower())
        tokens = re.findall(r"[a-z0-9_]{2,}", text)
        tokens.extend(re.findall(r"[\u4e00-\u9fff]", text))

        for size in (2, 3, 4):
            tokens.extend(text[i : i + size] for i in range(max(0, len(text) - size + 1)))
        return tokens
