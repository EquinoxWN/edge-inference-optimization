"""Round-to-nearest weight-only quantization of linear layers, written out so it can be tested.

int8: one symmetric scale per output channel, values in [-127, 127].
int4: one symmetric scale per group of 32 input weights, values in [-7, 7], two per byte.
Weights are stored quantized and dequantized to the activation dtype in each forward pass, so
this measures the accuracy and weight-memory effect of the format, not a speed-up (that needs
low-bit kernels, which the bitsandbytes variants provide).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional

INT4_GROUP = 32


def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric int8: returns (int8 weights, float16 scales)."""
    w = w.float()
    scale = (w.abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-12)
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(1).half()


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_int8, in float32."""
    return q.float() * scale.float().unsqueeze(1)


def quantize_int4(w: torch.Tensor, group: int = INT4_GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-wise symmetric int4 packed two per byte: returns (uint8 [out, in/2], float16 [out, in/group])."""
    out_f, in_f = w.shape
    if in_f % group or group % 2:
        raise ValueError(f"input size {in_f} must be a multiple of an even group size {group}")
    g = w.float().reshape(out_f, in_f // group, group)
    scale = (g.abs().amax(dim=2, keepdim=True) / 7).clamp(min=1e-12)
    q = torch.round(g / scale).clamp(-7, 7).to(torch.int16).reshape(out_f, in_f) + 8
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale.squeeze(2).half()


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack two signed 4-bit values per byte into int16 in [-8, 7]."""
    p = packed.to(torch.int16)
    low, high = (p & 0xF) - 8, ((p >> 4) & 0xF) - 8
    return torch.stack((low, high), dim=2).reshape(p.shape[0], p.shape[1] * 2)


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor, group: int = INT4_GROUP) -> torch.Tensor:
    """Inverse of quantize_int4, in float32."""
    q = unpack_int4(packed).float()
    out_f, in_f = q.shape
    return (q.reshape(out_f, in_f // group, group) * scale.float().unsqueeze(2)).reshape(out_f, in_f)


class QuantLinear(nn.Module):
    """A linear layer whose weight is stored as int8 or packed int4."""

    def __init__(self, linear: nn.Linear, bits: int) -> None:
        """Quantize an existing layer's weight; the bias stays in its dtype."""
        super().__init__()
        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        self.bits = bits
        self.in_features, self.out_features = linear.in_features, linear.out_features
        w = linear.weight.detach()
        q, s = quantize_int8(w) if bits == 8 else quantize_int4(w)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", s)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def weight_fp32(self) -> torch.Tensor:
        """The dequantized weight."""
        qw, sc = self.get_buffer("qweight"), self.get_buffer("scale")
        return dequantize_int8(qw, sc) if self.bits == 8 else dequantize_int4(qw, sc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize to the input dtype and apply the layer."""
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return functional.linear(x, self.weight_fp32().to(x.dtype), bias)

    def weight_bytes(self) -> int:
        """Bytes of the quantized weight and its scales."""
        return int(self.get_buffer("qweight").numel()) + 2 * int(self.get_buffer("scale").numel())


def replace_linears(model: nn.Module, bits: int, skip: Iterable[str] = ("classifier", "pre_classifier")) -> int:
    """Swap every nn.Linear whose name does not end in a skipped name; return how many were swapped."""
    skipped = tuple(skip)
    targets = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Linear) and name.split(".")[-1] not in skipped]
    for name, mod in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, QuantLinear(mod, bits))
    return len(targets)


def weight_bytes(model: nn.Module) -> int:
    """Bytes of all parameters and buffers, counting quantized layers at their stored size."""
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        total += t.numel() * t.element_size()
    return int(total)
