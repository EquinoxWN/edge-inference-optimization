"""Round-to-nearest int8 and int4 quantization."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from edge_inference_optimization.quant import (
    QuantLinear,
    dequantize_int4,
    dequantize_int8,
    quantize_int4,
    quantize_int8,
    replace_linears,
    unpack_int4,
    weight_bytes,
)


def test_int8_error_is_at_most_half_a_step() -> None:
    torch.manual_seed(1)
    w = torch.randn(16, 48) * torch.linspace(0.01, 3, 16).unsqueeze(1)
    q, s = quantize_int8(w)
    assert q.dtype == torch.int8
    assert s.dtype == torch.float16
    assert s.shape == (16,)
    assert int(q.abs().max()) == 127
    err = (dequantize_int8(q, s) - w).abs()
    # half a quantization step, plus the float16 rounding of the scale
    assert bool((err <= (s.float().unsqueeze(1) / 2) + w.abs() * 1e-3).all())


def test_int4_packing_round_trips_every_value() -> None:
    vals = torch.arange(-8, 8, dtype=torch.int16).repeat(4, 2)  # every nibble value
    nib = (vals + 8).to(torch.int16)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    assert torch.equal(unpack_int4(packed), vals)


def test_int4_groups_and_error() -> None:
    torch.manual_seed(2)
    w = torch.randn(8, 64)
    w[:, 32:] *= 100  # a second group with a much larger range keeps its own scale
    packed, s = quantize_int4(w, 32)
    assert packed.shape == (8, 32)
    assert packed.dtype == torch.uint8
    assert s.shape == (8, 2)
    err = (dequantize_int4(packed, s, 32) - w).abs().reshape(8, 2, 32)
    assert bool((err <= s.float().unsqueeze(2) / 2 + 1e-2 * w.abs().reshape(8, 2, 32)).all())
    assert float(err[:, 0].max()) < 0.5  # the small group is not ruined by the large one
    with pytest.raises(ValueError, match="multiple"):
        quantize_int4(torch.randn(4, 30), 32)


@pytest.mark.parametrize(("bits", "tol"), [(8, 0.02), (4, 0.25)])
def test_quant_linear_matches_float_layer(bits: int, tol: float) -> None:
    torch.manual_seed(3)
    lin = nn.Linear(64, 32)
    x = torch.randn(5, 64)
    q = QuantLinear(lin, bits)
    with torch.no_grad():
        rel = float((q(x) - lin(x)).norm() / lin(x).norm())
    assert rel < tol
    assert q.weight_bytes() == (32 * 64 if bits == 8 else 32 * 32) + 2 * (32 if bits == 8 else 32 * 2)
    with pytest.raises(ValueError, match="bits"):
        QuantLinear(lin, 3)


def test_replace_linears_skips_the_head_and_shrinks_weights() -> None:
    model = nn.Sequential()
    model.add_module("body", nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)))
    model.add_module("classifier", nn.Linear(64, 2))
    before = weight_bytes(model)
    assert replace_linears(model, 8, ["classifier"]) == 2
    assert isinstance(model.get_submodule("body.0"), QuantLinear)
    assert isinstance(model.get_submodule("classifier"), nn.Linear)
    assert weight_bytes(model) < before / 2
    assert model(torch.randn(3, 64)).shape == (3, 2)
