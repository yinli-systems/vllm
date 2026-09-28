# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for the byte footprint of floating-point operand tiles."""

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    triton_scaled_mm,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(17, 192, 256), (33, 512, 512), (129, 128, 128)])
@pytest.mark.parametrize("transpose_weight", [False, True])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_scaled_mm_wide_operand_footprint(
    dtype, shape, transpose_weight, per_channel, with_bias
):
    # The unmodified one-byte heuristic requests 128 KiB for several of these
    # shapes with FP16/BF16 operands, exceeding SM89/SM120's per-block limit.
    m, n, k = shape
    torch.manual_seed(20260928)
    a = (torch.rand((m, k), device="cuda", dtype=dtype) - 0.5).contiguous()
    b = (torch.rand((n, k), device="cuda", dtype=dtype) - 0.5).T
    if not transpose_weight:
        b = b.contiguous()
    scale_a = torch.rand((m if per_channel else 1, 1), device="cuda") + 0.25
    scale_b = torch.rand((n if per_channel else 1, 1), device="cuda") + 0.25
    bias = (
        torch.randn((n,), device="cuda", dtype=torch.bfloat16) * 0.01
        if with_bias
        else None
    )

    def run():
        return triton_scaled_mm(
            a, b, scale_a, scale_b, torch.bfloat16, bias=bias, use_td=False
        )

    # Match the kernel's documented epilogue order: scale, output cast, bias.
    reference = (a.double() @ b.double()) * scale_a.double() * scale_b.T.double()
    reference = reference.to(torch.bfloat16)
    if bias is not None:
        reference = reference + bias
    actual = run()
    torch.testing.assert_close(actual, reference, rtol=0.016, atol=0.001)
    relative_l2 = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative_l2.item() < 0.003

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = run()
    graph.replay()
    torch.testing.assert_close(replay_output, actual, rtol=0, atol=0)
