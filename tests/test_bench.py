"""Variants end to end on a tiny random model, CPU only."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from edge_inference_optimization import bench
from edge_inference_optimization.data import Example

CPU = torch.device("cpu")
FAST = bench.Settings(max_len=16, eval_batch=8, latency_requests=10, latency_warmup=2, throughput_batch=4, throughput_repeats=1)


def test_cpu_variants_measure_and_compare(tiny_model_dir: Path, examples: list[Example]) -> None:
    results = bench.run(tiny_model_dir, examples, ["fp32", "int8-rtn", "int4-rtn"], CPU, FAST)
    fp32, int8, int4 = results
    assert fp32.total == len(examples) == fp32.correct + (len(examples) - fp32.correct)
    assert fp32.flips is None
    assert int8.flips is not None
    assert int4.flips is not None
    assert int8.quantized_layers == 12  # 2 layers x (q, k, v, out, ffn1, ffn2); the head is kept
    assert int8.weight_mib < fp32.weight_mib
    assert int4.weight_mib < int8.weight_mib
    assert int8.agreement is not None
    assert int8.agreement >= 0.9
    for r in results:
        assert r.p50_ms is not None
        assert r.p95_ms is not None
        assert r.p50_ms <= r.p95_ms
        assert r.throughput is not None
        assert r.throughput > 0
        assert r.wh_per_1k is None
        assert r.peak_vram_mib is None
        assert "predictions" not in r.to_dict()


def test_same_variant_twice_gives_identical_predictions(tiny_model_dir: Path, examples: list[Example]) -> None:
    a, b = bench.run(tiny_model_dir, examples, ["fp32", "fp32"], CPU, FAST, speed=False)
    assert a.predictions == b.predictions
    assert b.flips == 0
    assert b.agreement == 1.0
    assert a.p50_ms is None


def test_variant_checks(tiny_model_dir: Path) -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        bench.load_variant("int3", tiny_model_dir, CPU)
    for name in bench.GPU_ONLY:
        with pytest.raises(ValueError, match="CUDA"):
            bench.load_variant(name, tiny_model_dir, CPU)


def test_throughput_energy_from_counter(tiny_model_dir: Path, examples: list[Example]) -> None:
    from transformers import AutoTokenizer

    model, _ = bench.load_variant("fp32", tiny_model_dir, CPU)
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    readings = iter([1_000_000, 1_000_000 + 3_600])  # 3.6 J over the measured window
    tput, rounds, wh = bench.throughput(model, tok, [e.text for e in examples], FAST, CPU, lambda: next(readings))
    assert tput > 0
    assert len(rounds) == FAST.throughput_repeats
    assert wh == pytest.approx(3_600 / 3.6e6 / len(examples) * 1000)
