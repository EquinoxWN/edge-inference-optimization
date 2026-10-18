"""The held-out set: GLUE SST-2 validation sentences with their sentiment labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

MAX_EXAMPLES = 100_000
MAX_CHARS = 2_000


@dataclass(frozen=True)
class Example:
    """One sentence and its label (0 negative, 1 positive)."""

    text: str
    label: int


def load_parquet(path: Path, limit: int | None = None) -> list[Example]:
    """Read sentence and label columns, refusing anything outside the expected schema."""
    table = pq.read_table(path, columns=["sentence", "label"])
    if table.num_rows > MAX_EXAMPLES:
        raise ValueError(f"{path.name}: {table.num_rows} rows, more than {MAX_EXAMPLES}")
    out: list[Example] = []
    for i, (text, label) in enumerate(zip(table.column("sentence").to_pylist(), table.column("label").to_pylist(), strict=True)):
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
            raise ValueError(f"{path.name}: row {i}: sentence must be 1..{MAX_CHARS} characters")
        if label not in (0, 1):
            raise ValueError(f"{path.name}: row {i}: label {label!r} is not 0 or 1")
        out.append(Example(text, int(label)))
        if limit is not None and len(out) >= limit:
            break
    return out
