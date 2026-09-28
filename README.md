# ParaCloud GPU upstream milestone — 2026-09-28

This is executed GPU evidence, not a claim that all seven issues are fixed.

## Published code

- vLLM PR 58953, draft: https://github.com/vllm-project/vllm/pull/58953
  Commit b2d6161e0f0cbe1e5d1df6f855f43d1e377a654a.
- SGLang PR 41498, draft: https://github.com/sgl-project/sglang/pull/41498
  Commit 82c2d781fe4ca2c1b5e14a98127d3c3a33ff917a.

## Interpretation

XQA: 20/20 real-CUDA comparison cases, six actual 1-to-0 host-copy cases,
plus 12/12 public GPU tests with poisoned padding and wrong-length negative
controls. The installed vLLM 0.29 equivalent port used an explicit shared
SM120 local-JIT availability prerequisite; this is disclosed in the PR and
included here. Complete current-main integration CI is not claimed.

Full-model Qwen3.5-4B MTP control: all 10 fixed-request output hashes match,
but 120 routing observations per arm were all causal. This workload does
not exercise the changed non-causal branch. Do NOT call its timing noise
an end-to-end speedup or a resolution of issue 54919.

Persistent GEMM: 108/108 operator cases and 18/18 public GPU pytest cases
per 4090/5090, followed by three fresh deterministic model processes per
GPU. Per-GPU output/logprob hashes agree across processes. Stage-one
controls may be faster than the proposed minimal two-stage capacity fix;
no universal throughput-maximization claim is made.

B12X: 32/32 prespecified numerical-gate cases, including wrong-output
negative controls and changing-input graph replay. The reference matches
the chosen precision; it is not a model-quality benchmark. Full-model
persistent per-layer wrapper allocation failed with OOM. Existing
FlashInfer PR 4603 already introduces shared workspaces; do not claim that
idea as new. The SM120 W4A16 kernel dependency fix is credited to PR 5195.

FlashInfer scoped prefill: four independent serving A/B pairs are included.
The small measured end-to-end effect is below the preregistered 3% promotion
target. The separate INT8 candidate with held-out cold-cache geomean
0.8640517526649508 remains rejected.

## Reproduction and raw data

`src/` contains executed harnesses; the Slurm scripts record environment
setup and command paths. `runs/` includes per-case numerical observations,
raw timing samples, JUnit reports, commands and exit receipts. `research/`
records preregistration and source provenance. Only machine/account path
identifiers are redacted to placeholders. `MANIFEST.json` records original
and distributed-byte SHA256 values. Full unredacted private logs remain
in the owner's original ParaCloud campaign directories.

Do not treat repeated timings within one process as independent experiment
replications. The FlashInfer serving interval uses four paired-server log
ratios, Student t with 3 degrees of freedom, not all request timings as N.

Files copied from upstream retain their own licenses; generated harnesses
and reports are provided under Apache-2.0. No model weights are distributed.

## Download integrity

`gpu_pr_milestone_20260928_v1.zip`: SHA256 `4106fe27e57cc763bb9be675dcab454f6c381b7862965c26528b728e89ad9200`.
