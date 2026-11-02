"""Pinned downloads: the model and the held-out set, each file checked by SHA-256."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

HF = "https://huggingface.co"
MAX_DOWNLOAD = 512 * 1024 * 1024


@dataclass(frozen=True)
class Pinned:
    """One file at a fixed URL with its expected size and SHA-256."""

    name: str
    url: str
    size: int
    sha256: str


MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
DATASET_ID = "nyu-mll/glue"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"


def _model(name: str, size: int, sha: str) -> Pinned:
    """A model file at the pinned revision."""
    return Pinned(name, f"{HF}/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}", size, sha)


MODEL_FILES = (
    _model("config.json", 629, "582122c8f414793d131e10022ce9ba04e3811a9da6389137ee2f18665b4f4d15"),
    _model("vocab.txt", 231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    _model("tokenizer_config.json", 48, "5ab9097b4149371c5fd52b2d6e26cb6f9c07c0d19fcfdda895b1adad6b57c3e0"),
    _model("model.safetensors", 267832558, "7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa"),
)
VALIDATION = Pinned(
    "validation.parquet",
    f"{HF}/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/sst2/validation-00000-of-00001.parquet",
    72819,
    "a1371f3b3a7b0bcefa8388799a9359dc3ce76c349cc0079507a7991364fd2a9b",
)


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, p: Pinned) -> None:
    """Raise unless the file has the pinned size and hash."""
    size = path.stat().st_size
    if size != p.size:
        raise ValueError(f"{path.name}: size {size}, expected {p.size}")
    got = sha256_of(path)
    if got != p.sha256:
        raise ValueError(f"{path.name}: SHA-256 {got}, expected {p.sha256}")


def copy_limited(src: BinaryIO, dst: BinaryIO, size: int) -> None:
    """Copy at most size bytes; a longer body is refused instead of filling the disk."""
    left = size
    while chunk := src.read(min(1 << 20, left + 1)):
        left -= len(chunk)
        if left < 0:
            raise ValueError(f"download larger than the pinned {size} bytes")
        dst.write(chunk)


def fetch(p: Pinned, dest_dir: Path) -> Path:
    """Download over HTTPS once, then check size and hash; a bad file is deleted."""
    if not p.url.startswith("https://") or p.size > MAX_DOWNLOAD:
        raise ValueError(f"{p.name}: refusing {p.url}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / p.name
    if not out.exists():
        part = out.with_suffix(out.suffix + ".part")
        req = urllib.request.Request(p.url, headers={"User-Agent": "edge-inference-optimization"})  # noqa: S310 (https only, checked above)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as f:  # noqa: S310 (https only, checked above)
                copy_limited(resp, f, p.size)
        except BaseException:
            part.unlink(missing_ok=True)  # never leave a partial download behind
            raise
        part.replace(out)
    try:
        verify(out, p)
    except ValueError:
        out.unlink()
        raise
    return out
