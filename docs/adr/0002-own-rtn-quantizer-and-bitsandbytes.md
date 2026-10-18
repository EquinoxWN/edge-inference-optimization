# ADR 0002: Measure an own round-to-nearest quantizer next to bitsandbytes

- **Status:** Accepted

## Context

M1 must quantize the model to int8 and int4 and measure the accuracy lost. Libraries such as
bitsandbytes do this in one line with fast GPU kernels, but their formats (LLM.int8 with fp16
outlier columns, NF4 with a non-uniform code book) differ from plain int8 and int4, and their
internals are not visible in this repository. A hand-written quantizer is transparent and
testable, but without low-bit kernels it can only dequantize weights before each matrix multiply.

## Decision

Measure both. `quant.py` implements round-to-nearest int8 (per output channel) and int4 (groups of
32, packed two per byte) with unit tests on the error bound and the packing; the same 36 linear
layers are also loaded through bitsandbytes as LLM.int8 and NF4. Embeddings and the
classification head stay in floating point in every variant, so the comparison is about the
transformer's linear layers only.

## Consequences

- Accuracy and memory of each format are measured directly: int8 RTN lost 0.11 points (1 flipped
  prediction), int4 RTN 0.92 points (14 flips), LLM.int8 and NF4 0.23 points each (4 and 18 flips).
- Speed of the RTN variants reflects the dequantize-per-call design (about 1,400 to 1,600
  sentences per second against 2,100 to 2,250 for fp16), not int8 hardware; this is stated wherever
  the numbers appear.
- LLM.int8 was the slowest variant (750 to 880 sentences per second, p50 28 to 30 ms per request):
  its mixed int8 and float16 decomposition did not pay off on a model this small (not profiled further).
- Two quantizers double the variants to maintain; worth it while the formats are being compared,
  and the RTN code is the reference that calibrated methods (GPTQ, AWQ) would be measured against.
