"""Load each model variant, then measure accuracy, agreement, memory, latency and energy."""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

from .data import Example
from .quant import replace_linears, weight_bytes

MIB = 1024 * 1024
GPU_ONLY = ("fp16", "int8-bnb", "nf4-bnb")
VARIANTS = ("fp32", "fp16", "int8-rtn", "int4-rtn", "int8-bnb", "nf4-bnb")
SKIP = ["pre_classifier", "classifier"]


@dataclass
class Settings:
    """How each variant is measured."""

    max_len: int = 128
    eval_batch: int = 64
    latency_requests: int = 300
    latency_warmup: int = 30
    throughput_batch: int = 32
    throughput_repeats: int = 3
    gpu_warmup_seconds: float = 20.0


@dataclass
class VariantResult:
    """Everything measured for one variant."""

    name: str
    device: str
    accuracy: float
    correct: int
    total: int
    weight_mib: float
    quantized_layers: int = 0
    agreement: float | None = None
    flips: int | None = None
    peak_vram_mib: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    throughput: float | None = None
    throughput_rounds: list[float] = field(default_factory=list)
    wh_per_1k: float | None = None
    sm_clock_mhz: int | None = None
    predictions: list[int] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form without the per-example predictions."""
        out = asdict(self)
        out.pop("predictions")
        return out


def load_variant(name: str, model_dir: Path, device: torch.device) -> tuple[nn.Module, int]:
    """Load the model in one of VARIANTS; returns it in eval mode with the quantized-layer count."""
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}")
    if name in GPU_ONLY and device.type != "cuda":
        raise ValueError(f"{name} needs a CUDA GPU")
    if name.endswith("-bnb"):
        from transformers import BitsAndBytesConfig

        q = (
            BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=SKIP)  # type: ignore[no-untyped-call]
            if name == "int8-bnb"
            else BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_skip_modules=SKIP,
            )
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, quantization_config=q, device_map={"": device.index or 0}, dtype=torch.float16
        )
        layers = sum(1 for m in model.modules() if type(m).__name__ in ("Linear8bitLt", "Linear4bit"))
        return model.eval(), layers
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32)
    layers = 0
    if name.endswith("-rtn"):
        layers = replace_linears(model, 8 if name.startswith("int8") else 4, SKIP)
    if device.type == "cuda" and name != "fp32":
        model = model.half()
    return model.to(device).eval(), layers


def _sync(device: torch.device) -> None:
    """Wait for queued GPU work."""
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def predict(model: nn.Module, tok: PreTrainedTokenizerBase, texts: Sequence[str], s: Settings, device: torch.device) -> list[int]:
    """Labels for texts, in batches."""
    out: list[int] = []
    for i in range(0, len(texts), s.eval_batch):
        enc = tok(list(texts[i : i + s.eval_batch]), padding=True, truncation=True, max_length=s.max_len, return_tensors="pt").to(device)
        out.extend(model(**enc).logits.argmax(dim=-1).tolist())
    return out


@torch.inference_mode()
def latency_ms(
    model: nn.Module, tok: PreTrainedTokenizerBase, texts: Sequence[str], s: Settings, device: torch.device
) -> tuple[float, float]:
    """p50 and p95 of one-request latency (tokenize, forward, argmax)."""
    times: list[float] = []
    for i in range(s.latency_warmup + s.latency_requests):
        text = texts[i % len(texts)]
        _sync(device)
        t0 = time.perf_counter()
        enc = tok([text], truncation=True, max_length=s.max_len, return_tensors="pt").to(device)
        int(model(**enc).logits.argmax(dim=-1)[0])
        _sync(device)
        if i >= s.latency_warmup:
            times.append((time.perf_counter() - t0) * 1000)
    q = statistics.quantiles(times, n=20, method="inclusive")
    return statistics.median(times), q[18]


@torch.inference_mode()
def throughput(
    model: nn.Module,
    tok: PreTrainedTokenizerBase,
    texts: Sequence[str],
    s: Settings,
    device: torch.device,
    energy: Callable[[], int | None],
) -> tuple[float, list[float], float | None]:
    """Median sentences per second over the rounds, each round's rate, and GPU Wh per 1,000 sentences."""
    batches = [list(texts[i : i + s.throughput_batch]) for i in range(0, len(texts), s.throughput_batch)]
    encoded = [tok(b, padding=True, truncation=True, max_length=s.max_len, return_tensors="pt").to(device) for b in batches]
    model(**encoded[0])  # warm-up
    _sync(device)
    mj0 = energy()
    rounds: list[float] = []
    for _ in range(s.throughput_repeats):
        t0 = time.perf_counter()
        for enc in encoded:
            model(**enc).logits.argmax(dim=-1)
        _sync(device)
        rounds.append(len(texts) / (time.perf_counter() - t0))
    mj1 = energy()
    n = len(texts) * s.throughput_repeats
    wh = None if mj0 is None or mj1 is None else (mj1 - mj0) / 3.6e6 / n * 1000
    return statistics.median(rounds), rounds, wh


def nvml_sm_clock() -> Callable[[], int | None]:
    """A function returning the GPU's current SM clock in MHz, or None if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
    except Exception:  # no driver or no GPU
        return lambda: None
    return lambda: int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))


def gpu_warmup(device: torch.device, seconds: float) -> None:
    """Keep the GPU busy so its clocks leave the idle state before anything is timed."""
    if device.type != "cuda" or seconds <= 0:
        return
    a = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        for _ in range(20):
            a = torch.tanh(a @ a)
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def nvml_energy() -> Callable[[], int | None]:
    """A function returning the GPU's total energy counter in mJ, or None if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except Exception:  # no driver, no GPU or no counter
        return lambda: None
    return lambda: int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))


def evaluate_variant(
    name: str,
    model_dir: Path,
    examples: Sequence[Example],
    device: torch.device,
    s: Settings,
    reference: Sequence[int] | None = None,
    energy: Callable[[], int | None] = lambda: None,
    speed: bool = True,
    clock: Callable[[], int | None] = lambda: None,
) -> VariantResult:
    """Load one variant, measure it, free it."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model, layers = load_variant(name, model_dir, device)
    # bitsandbytes layers keep packed weights outside ordinary parameters; transformers counts them.
    footprint = int(getattr(model, "get_memory_footprint")()) if name.endswith("-bnb") else weight_bytes(model)  # noqa: B009
    texts = [e.text for e in examples]
    preds = predict(model, tok, texts, s, device)
    correct = sum(int(p == e.label) for p, e in zip(preds, examples, strict=True))
    r = VariantResult(
        name, str(device), correct / len(examples), correct, len(examples), round(footprint / MIB, 2), layers, predictions=preds
    )
    if device.type == "cuda":
        r.peak_vram_mib = round(torch.cuda.max_memory_allocated() / MIB, 1)
    if reference is not None:
        r.flips = sum(int(a != b) for a, b in zip(preds, reference, strict=True))
        r.agreement = 1 - r.flips / len(preds)
    if speed:
        p50, p95 = latency_ms(model, tok, texts, s, device)
        r.p50_ms, r.p95_ms = round(p50, 2), round(p95, 2)
        tput, rounds, wh = throughput(model, tok, texts, s, device, energy)
        r.throughput = round(tput, 1)
        r.throughput_rounds = [round(x, 1) for x in rounds]
        r.sm_clock_mhz = clock()
        r.wh_per_1k = None if wh is None else round(wh, 4)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return r


def run(
    model_dir: Path, examples: Sequence[Example], variants: Sequence[str], device: torch.device, s: Settings, speed: bool = True
) -> list[VariantResult]:
    """Measure every variant; the first one is the reference for agreement."""
    energy = nvml_energy() if device.type == "cuda" else (lambda: None)
    clock = nvml_sm_clock() if device.type == "cuda" else (lambda: None)
    if speed:
        gpu_warmup(device, s.gpu_warmup_seconds)
    results: list[VariantResult] = []
    for name in variants:
        ref = results[0].predictions if results else None
        results.append(evaluate_variant(name, model_dir, examples, device, s, ref, energy, speed, clock))
    return results
