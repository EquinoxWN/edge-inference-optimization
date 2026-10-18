"""Command line: eio prepare | run | show | purge."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import env

DEFAULT_OUT = "docs/results/m1-run.json"


def paths() -> tuple[Path, Path]:
    """Model and data folders inside .tmp."""
    t = env.tmp_dir()
    return t / "models" / "distilbert-sst2", t / "data"


def prepare() -> tuple[Path, Path]:
    """Download the pinned model and held-out set (SHA-256 checked)."""
    from .sources import MODEL_FILES, VALIDATION, fetch

    model_dir, data_dir = paths()
    for p in MODEL_FILES:
        fetch(p, model_dir)
    return model_dir, fetch(VALIDATION, data_dir)


def purge() -> list[str]:
    """Delete the downloaded model and data; return what was removed."""
    removed = []
    for p in paths():
        if p.exists():
            shutil.rmtree(env.ensure_inside(p, env.tmp_dir()))
            removed.append(str(p))
    return removed


def table(results: Sequence[dict[str, Any]]) -> str:
    """Markdown table of a run."""

    def f(v: Any, fmt: str) -> str:
        """Format a value or a dash."""
        return "-" if v is None else format(v, fmt)

    base = results[0]["accuracy"]
    lines = [
        "| Variant | Accuracy | Change vs first (pp) | Flipped predictions | Weights (MiB) | Peak VRAM (MiB) | p50 ms | p95 ms | Sentences/s | GPU Wh per 1k | SM MHz |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r['name']} | {r['accuracy'] * 100:.2f}% ({r['correct']}/{r['total']}) | {(r['accuracy'] - base) * 100:+.2f} | "
        f"{f(r['flips'], 'd')} | {r['weight_mib']:.1f} | {f(r['peak_vram_mib'], '.0f')} | {f(r['p50_ms'], '.2f')} | "
        f"{f(r['p95_ms'], '.2f')} | {f(r['throughput'], '.0f')} | {f(r['wh_per_1k'], '.4f')} | {f(r.get('sm_clock_mhz'), 'd')} |"
        for r in results
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    """Measure the variants and write the results JSON."""
    import torch

    from . import bench
    from .data import load_parquet
    from .sources import MODEL_ID, MODEL_REVISION, VALIDATION

    model_dir, data_file = prepare()
    examples = load_parquet(data_file, args.limit)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    variants = args.variants or [v for v in bench.VARIANTS if device.type == "cuda" or v not in bench.GPU_ONLY]
    torch.manual_seed(0)
    try:
        results = bench.run(model_dir, examples, variants, device, bench.Settings(), speed=not args.no_speed)
    finally:
        if not args.keep_model:
            purge()
    meta = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dataset": "GLUE SST-2 validation",
        "dataset_sha256": VALIDATION.sha256,
        "examples": len(examples),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "settings": asdict(bench.Settings()),
        "model_deleted_after_run": not args.keep_model,
    }
    doc: dict[str, Any] = {"meta": meta, "variants": [r.to_dict() for r in results]}
    out = env.ensure_inside(Path(args.out), env.repo_root())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(table(doc["variants"]))
    print(f"wrote {out.relative_to(env.repo_root())}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch."""
    env.configure()
    p = argparse.ArgumentParser(prog="eio", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="download the pinned model and held-out set")
    r = sub.add_parser("run", help="measure every variant, then delete the model")
    r.add_argument("--out", default=DEFAULT_OUT)
    r.add_argument("--variants", nargs="+", choices=["fp32", "fp16", "int8-rtn", "int4-rtn", "int8-bnb", "nf4-bnb"])
    r.add_argument("--limit", type=int, help="use only the first N held-out sentences")
    r.add_argument("--cpu", action="store_true", help="run on the CPU even if a GPU is present")
    r.add_argument("--no-speed", action="store_true", help="skip latency, throughput and energy")
    r.add_argument("--keep-model", action="store_true", help="do not delete the model afterwards")
    s = sub.add_parser("show", help="print a results file as a table")
    s.add_argument("file", nargs="?", default=DEFAULT_OUT)
    sub.add_parser("purge", help="delete the downloaded model and data")
    args = p.parse_args(argv)
    if args.cmd == "prepare":
        model_dir, data_file = prepare()
        print(f"model in {model_dir}, held-out set {data_file}")
    elif args.cmd == "run":
        if args.limit is not None and args.limit < 1:
            p.error("--limit must be positive")
        return run(args)
    elif args.cmd == "show":
        doc = json.loads(Path(args.file).read_text(encoding="utf-8"))
        print(table(doc["variants"]))
    else:
        print("removed: " + (", ".join(purge()) or "nothing"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
