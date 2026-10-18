"""Data loading, pinned downloads, environment containment and the CLI."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from edge_inference_optimization import cli, env
from edge_inference_optimization.data import load_parquet
from edge_inference_optimization.sources import MODEL_FILES, VALIDATION, Pinned, fetch, sha256_of, verify


def write_parquet(path: Path, sentences: Sequence[object], labels: Sequence[object]) -> Path:
    """A parquet file with the SST-2 columns."""
    pq.write_table(pa.table({"sentence": sentences, "label": labels, "idx": list(range(len(labels)))}), path)
    return path


def test_load_parquet_validates_rows(tmp_path: Path) -> None:
    good = write_parquet(tmp_path / "g.parquet", ["fine film", "dull"], [1, 0])
    assert [(e.text, e.label) for e in load_parquet(good)] == [("fine film", 1), ("dull", 0)]
    assert len(load_parquet(good, limit=1)) == 1
    for sentences, labels in ([" "], [1]), (["ok"], [2]), (["x" * 2001], [0]):
        with pytest.raises(ValueError, match="row 0"):
            load_parquet(write_parquet(tmp_path / "b.parquet", sentences, labels))


def test_verify_and_fetch_refuse_bad_files(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    good = Pinned("a.bin", "https://example.invalid/a.bin", 5, sha256_of(f))
    verify(f, good)
    with pytest.raises(ValueError, match="SHA-256"):
        fetch(Pinned("a.bin", good.url, 5, "0" * 64), tmp_path)  # file exists, so no download
    assert not f.exists()  # a file that fails the check is deleted
    with pytest.raises(ValueError, match="refusing"):
        fetch(Pinned("b", "http://example.invalid/b", 1, "0" * 64), tmp_path)
    f.write_bytes(b"hi")
    with pytest.raises(ValueError, match="size"):
        verify(f, good)


def test_pins_are_complete() -> None:
    for p in (*MODEL_FILES, VALIDATION):
        assert p.url.startswith("https://huggingface.co/")
        assert len(p.sha256) == 64
        assert p.size > 0
        assert int(p.sha256, 16) != 0


def test_configure_keeps_caches_inside(tmp_path: Path) -> None:
    applied = env.configure(tmp_path)
    for var in ("HF_HOME", "TORCH_HOME", "TMP", "CUDA_CACHE_PATH"):
        assert Path(applied[var]).is_relative_to(tmp_path)
        assert os.environ[var] == applied[var]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    env.configure()  # restore the repository's own paths for the other tests
    with pytest.raises(ValueError, match="outside"):
        env.ensure_inside(tmp_path / ".." / "..", tmp_path)


def test_show_prints_a_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    row = {
        "name": "fp32",
        "accuracy": 0.9,
        "correct": 9,
        "total": 10,
        "flips": None,
        "weight_mib": 255.4,
        "peak_vram_mib": None,
        "p50_ms": 1.234,
        "p95_ms": 2.0,
        "throughput": 100.0,
        "wh_per_1k": None,
    }
    second = row | {"name": "int8-rtn", "accuracy": 0.8, "correct": 8, "flips": 1}
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"meta": {}, "variants": [row, second]}), encoding="utf-8")
    assert cli.main(["show", str(f)]) == 0
    out = capsys.readouterr().out
    assert "| fp32 | 90.00% (9/10) | +0.00 | - | 255.4 | - | 1.23 |" in out
    assert "| int8-rtn | 80.00% (8/10) | -10.00 | 1 |" in out


def test_purge_only_touches_tmp(capsys: pytest.CaptureFixture[str]) -> None:
    model_dir, data_dir = cli.paths()
    assert model_dir.is_relative_to(env.tmp_dir())
    assert data_dir.is_relative_to(env.tmp_dir())
    assert cli.main(["purge"]) == 0
    assert "removed" in capsys.readouterr().out
