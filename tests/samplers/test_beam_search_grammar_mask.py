# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grammar masks in offline beam search: unpacking and allowed-token lookup."""

from types import SimpleNamespace

import pytest
import torch

from vllm.entrypoints.generate.beam_search import offline
from vllm.entrypoints.generate.beam_search.offline import (
    _MAX_NUM_ALLOWED_TOKEN_IDS,
    _AllowedTokens,
    _bitmask_to_allowed,
)
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams


def _pack(allowed: set[int], vocab_size: int, extra_words: int = 0) -> torch.Tensor:
    """Pack token IDs the way the structured-output backends do: token i is
    bit i % 32 of int32 word i // 32."""
    words = [0] * ((vocab_size + 31) // 32 + extra_words)
    for i in allowed:
        words[i // 32] |= 1 << (i % 32)
    return torch.tensor(
        [w - (1 << 32) if w >= 1 << 31 else w for w in words], dtype=torch.int32
    )


def _reference(row: torch.Tensor, vocab_size: int) -> list[bool]:
    return [bool((int(row[i // 32]) >> (i % 32)) & 1) for i in range(vocab_size)]


@pytest.mark.parametrize("little_endian_fast_path", [True, False])
@pytest.mark.parametrize("vocab_size", [1, 31, 32, 33, 64, 1000])
@pytest.mark.parametrize(
    "allowed_fn",
    [
        lambda v: set(),
        lambda v: set(range(v)),
        lambda v: {0, v - 1},
        lambda v: {i for i in range(v) if i % 32 == 31},  # int32 sign bits
        lambda v: {i for i in range(v) if i % 7 == 3},
    ],
)
def test_bitmask_to_allowed_matches_bit_layout(
    monkeypatch, little_endian_fast_path, vocab_size, allowed_fn
):
    monkeypatch.setattr(offline, "_LITTLE_ENDIAN", little_endian_fast_path)
    allowed = allowed_fn(vocab_size)
    # Padding bits past the vocabulary and padding words must be ignored.
    row = _pack(allowed | {vocab_size, vocab_size + 5}, vocab_size, extra_words=2)
    assert _bitmask_to_allowed(row, vocab_size).tolist() == _reference(row, vocab_size)


def test_bitmask_to_allowed_handles_views_and_leaves_the_row_alone():
    vocab_size = 100
    packed = _pack({1, 33, 99}, vocab_size)
    wide = torch.zeros(packed.numel(), 2, dtype=torch.int32)
    wide[:, 0] = packed
    view = wide[:, 0]  # non-contiguous
    assert _bitmask_to_allowed(view, vocab_size).nonzero()[0].tolist() == [1, 33, 99]
    # The caller's tensor never shares storage with NumPy, so it stays
    # resizable and later refills are not observed by earlier results.
    result = _bitmask_to_allowed(packed, vocab_size)
    packed.zero_()
    packed.resize_(packed.numel() + 1)
    assert result.nonzero()[0].tolist() == [1, 33, 99]


def test_allowed_tokens_membership():
    allowed = _AllowedTokens(_bitmask_to_allowed(_pack({0, 5, 63}, 64), 64))
    assert [t for t in range(-2, 70) if t in allowed] == [0, 5, 63]


class _FakeGrammar:
    def __init__(self, allowed: set[int] | None):
        self.allowed = allowed

    def accept_tokens(self, request_id, tokens):
        return True

    def is_terminated(self):
        return self.allowed is None

    def fill_bitmask(self, bitmask, idx):
        bitmask[idx] = _pack(self.allowed, VOCAB)


VOCAB = 151_936


@pytest.mark.parametrize(
    ("allowed", "expect_engine_list"),
    [
        ({3, 70_000, 151_935}, True),  # sparse: the engine gets the list
        (set(range(_MAX_NUM_ALLOWED_TOKEN_IDS)), True),  # at the cap
        (set(range(_MAX_NUM_ALLOWED_TOKEN_IDS + 1)), False),  # past the cap
        (set(range(VOCAB)) - {7}, False),  # dense, as inside a JSON string
    ],
)
def test_build_beam_sampling_params(allowed, expect_engine_list):
    llm = LLM.__new__(LLM)
    llm.__dict__["model_config"] = SimpleNamespace(get_vocab_size=lambda: VOCAB)
    backend = SimpleNamespace(
        compile_grammar=lambda request_type, spec: _FakeGrammar(allowed)
    )
    beam = BeamSearchSequence(
        orig_prompt={"type": "token", "prompt_token_ids": [1, 2]},
        tokens=[1, 2, 9],
        logprobs=[],
    )
    bitmask = torch.zeros(1, (VOCAB + 31) // 32, dtype=torch.int32)

    [entry] = llm._build_beam_sampling_params(
        [beam],
        SamplingParams(logprobs=8, max_tokens=1),
        backend,
        ("json", "{}"),
        bitmask,
    )

    params, allowed_tokens = entry
    if expect_engine_list:
        assert params.allowed_token_ids == sorted(allowed)
    else:
        assert params.allowed_token_ids is None
    for token_id in (0, 3, 7, 1023, 1024, 70_000, VOCAB - 1, VOCAB, VOCAB + 1):
        assert (token_id in allowed_tokens) == (token_id in allowed)


@pytest.mark.parametrize("allowed", [set(), None])  # dead end, terminated
def test_build_beam_sampling_params_drops_finished_beams(allowed):
    llm = LLM.__new__(LLM)
    llm.__dict__["model_config"] = SimpleNamespace(get_vocab_size=lambda: VOCAB)
    backend = SimpleNamespace(
        compile_grammar=lambda request_type, spec: _FakeGrammar(allowed)
    )
    beam = BeamSearchSequence(
        orig_prompt={"type": "token", "prompt_token_ids": [1]},
        tokens=[1, 9],
        logprobs=[],
    )
    bitmask = torch.zeros(1, (VOCAB + 31) // 32, dtype=torch.int32)
    assert llm._build_beam_sampling_params(
        [beam],
        SamplingParams(logprobs=8, max_tokens=1),
        backend,
        ("json", "{}"),
        bitmask,
    ) == [None]
