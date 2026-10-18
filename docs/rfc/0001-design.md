# RFC 0001: edge-inference-optimization design

- **Status:** Accepted for M1
- **Author:** EquinoxWN
- **Created:** 2026-10-18

## Problem

Teams serve models on GPUs because that is how they were trained, then pay for GPU time on every
request. Smaller numeric formats (fp16, int8, int4) and later distillation and CPU runtimes can
cut that cost, but each step can also lose accuracy, and the loss is often assumed rather than
measured. Speed numbers on a shared laptop GPU are also easy to get wrong: clocks ramp up and
down, and the first model measured pays for it.

M1 sets the reference: a baseline served in plain PyTorch on the GPU, then int8 and int4
versions, each measured on the same held-out set for accuracy, changed predictions, weight
memory, latency, throughput and GPU energy.

**Users:** ML and platform engineers deciding how to serve a classifier cheaply, and people
learning what quantization costs and buys.

## Define: critical-to-quality measures (CTQs)

| CTQ | Target | Measured by |
|---|---|---|
| Accuracy loss reported per step | accuracy, change in points and flipped predictions vs the fp32 baseline for every variant on all 872 held-out sentences | `eio run` |
| int8 is safe | int8 loses at most 0.5 accuracy points | `eio run` |
| int4 cost is known | int4 loss measured, not assumed; accepted if within 1.5 points | `eio run` |
| Repeatable | accuracy identical across runs; throughput of each variant within 25% between two runs | two full runs |
| Quantizer is correct | int8 and int4 round-trip error within half a quantization step; packing exact; layer outputs close to float | unit tests |
| Pinned and clean | model and data pinned by revision and SHA-256; the model deleted after each run; caches in `.tmp` | `sources.py`, `cli.purge`, outside-folder check |

## Goals (M1)

- A baseline model is served in plain PyTorch on GPU to set the reference for accuracy, latency and cost.
- The model is quantized to int8 and int4, and the accuracy lost at each step is measured on a held-out set.

## Non-goals

- Distillation, ONNX export and graph optimisation (M2); ONNX Runtime on CPUs, llama.cpp and a
  dollar cost model (M3). M1 records GPU energy per 1,000 requests as the cost signal.
- Quantization-aware training or calibration-based methods (GPTQ, AWQ, SmoothQuant).
- An HTTP server: "served" here means the request path (tokenize, forward, argmax) timed one
  request at a time, plus batched throughput.

## Design

```
eio run
 ├─ fetch   model distilbert-base-uncased-finetuned-sst-2-english @714eb0fa (config, vocab, safetensors)
 │          + GLUE SST-2 validation parquet @bcdcba79, each checked by size and SHA-256
 ├─ warm    20 s of fp16 matmuls so the laptop GPU leaves its idle clocks
 ├─ per variant: load -> accuracy on 872 sentences (batch 64) -> flips vs fp32
 │               -> latency: 300 single requests after 30 warm-up (p50, p95)
 │               -> throughput: 3 rounds over 872 sentences at batch 32 (median), NVML energy
 └─ write docs/results/runs/*.json, print a table, delete the model and data
```

Variants (the embeddings and the classification head stay in floating point in every quantized
variant; the 36 linear layers of the 6 transformer blocks are quantized):

| Variant | Weights | Compute | Kernels |
|---|---|---|---|
| `fp32` (baseline) | float32 | float32 | PyTorch |
| `fp16` | float16 | float16 | PyTorch |
| `int8-rtn` | int8, one scale per output channel | float16 (dequantized per call) | own code |
| `int4-rtn` | int4, one scale per 32 inputs, packed two per byte | float16 (dequantized per call) | own code |
| `int8-bnb` | LLM.int8 (bitsandbytes) | int8 matmul with fp16 outliers | bitsandbytes |
| `nf4-bnb` | NF4 4-bit (bitsandbytes) | float16 | bitsandbytes |

The round-to-nearest (RTN) quantizer in `quant.py` is written out so the format is visible and
unit-tested; bitsandbytes provides production kernels for the same bit widths (ADR 0002).

## Alternatives considered

| Option | Why not (yet) |
|---|---|
| Only bitsandbytes | Real kernels, but the format is hidden behind a library; the own RTN quantizer makes the int8 and int4 formats and their error testable, and shows where the cost comes from. |
| PyTorch dynamic int8 quantization (`torch.ao`) | CPU only; the M1 baseline is on the GPU. It belongs with the CPU work in M2 and M3. |
| GPTQ or AWQ calibration | Lower int4 loss on large models, but needs calibration data and extra dependencies; RTN is the reference point they are compared against. |
| A large language model as the subject | The 6 GB GPU and the 872-example labelled set make a classifier the model whose accuracy can be measured exactly; llama.cpp for LLMs is M3. |
| Timing without warm-up, one round | Tried first: throughput differed by up to 4.6 times between runs for the same variant (see results). |

## Measurement plan

- Two full runs on the laptop (RTX 4050 Laptop GPU, 6 GB), all six variants, all 872 sentences.
- Unit tests in CI on a tiny random DistilBERT (CPU) for the pipeline and the quantizer.
- Results, root causes and controls in `docs/results/m1.md` (Measure, Analyze, Improve, Control).

## Milestones

- **M1:** A baseline model is served in plain PyTorch on GPU to set the reference for accuracy, latency and cost. The model is quantized to int8 and int4, and the accuracy lost at each step is measured on a held-out set.
- **M2:** A smaller student model is distilled from the original, trading a little accuracy for a large speed-up. The model is exported to ONNX, and graph optimization plus operator fusion streamline it for CPU.
- **M3:** ONNX Runtime serves it on commodity CPUs with tuned threading and dynamic batching; llama.cpp is compared for LLMs. A cost model turns latency and hardware price into cost per 1,000 requests, showing the CPU-vs-GPU trade-off.

## Risks and open questions

- The model was fine-tuned on SST-2 train; the validation split is held out from training but is
  the split its authors reported, so it is not unseen in the model-selection sense.
- 872 sentences: one sentence is 0.11 points; differences of a few flips are within noise.
- A shared laptop GPU: clocks and other work change speed numbers; the runs record the SM clock and
  three throughput rounds so this is visible. Energy includes the GPU's own idle draw.
- RTN variants dequantize in every forward pass, so their speed is a property of this code, not of
  int8 or int4 hardware support.
