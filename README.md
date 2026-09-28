# Corrected evidence snapshot v2

This version supersedes v1's SGLang commit metadata. The earlier strings
`82c2d781fe4ca2c1b5e14a98127d3c3a33ff917a` and
`6cbdfec13c465e4a549b5d465150d0031b43051b` are withdrawn: they could not be
verified. GitHub PR metadata, local Git and the remote branch agree on
head `8d624dd7ce3f77880eddcc0657366baedcfbdb15` and parent
`6cbdfec45308759009511894cccd8654c893392b`. Crucially, the exact production
and test bytes match the SHA256 receipts from GPU jobs 1632978/1632979.
The prior metadata is preserved in a clearly labeled audit file; raw
numerical observations were not changed.

Additional final-state records: `FINAL_AUDIT.json` and exact submitted
`patches/*-submitted.patch`. The SGLang PR is now ready for review; vLLM
remains draft. Neither is merged and completed upstream CI is not claimed.

The original MTP diagnostic's Python counter missed copies: the profiler
correlates five D2H operations within the length-read markers. Its zero-copy
interpretation is withdrawn in `evidence/54919-copy-trace-reviewed.json`.
The diagnostic candidate arm never ran; this is not a paired speedup result.

The remaining full-model B12X job 1633013 failed in our adapter (`quant_mode`
missing). Its attempted follow-up was blocked before execution by a tool
safety check; no new job was submitted. The 27B capacity-adjusted job1633018
hit the HTTP-server startup deadline and completed no prompt requests.
Neither failure is evidence of reproducing or fixing a CUDA IMA. At this
snapshot the user's Slurm queue is empty; see final job accounting.

---

# ParaCloud GPU upstream milestone — 2026-09-28

This is executed GPU evidence, not a claim that all seven issues are fixed.

## Published code

- vLLM PR 58953, draft: https://github.com/vllm-project/vllm/pull/58953
  Commit b2d6161e0f0cbe1e5d1df6f855f43d1e377a654a.
- SGLang PR 41498, ready for review (not merged): https://github.com/sgl-project/sglang/pull/41498
  Verified commit 8d624dd7ce3f77880eddcc0657366baedcfbdb15; source/test SHA256 match both completed GPU pytest jobs.

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

## Current download

Use `gpu_pr_milestone_20260928_v2.zip` (228 files; 238,836 bytes).
SHA256: `966d2327c78c6b471bc9af49f0a2d173bfe84c2757aa8b930f681729371cc39e`.
The v1 archive is retained for audit history and is superseded for commit metadata.
