from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kb import ingest


if __name__ == "__main__":
    result = ingest(reset=True)
    print(result)
