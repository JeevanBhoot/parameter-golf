# Parameter Golf CUDA/PyTorch Replication Program

## Mission

Your job is to replicate the strongest MLX quantization schemes on CUDA/PyTorch.

This is **not** a fresh quantization search.
Do not invent new schemes here.
Do not do reinvestment here.
Do not chase architecture changes here.

The job is:
1. Port the selected MLX quantization methods into `train_gpt.py`-style CUDA/PyTorch branches.
2. Confirm they run.
3. Confirm the logging, artifact accounting, and evaluation protocol match the MLX quick-track setup as closely as practical.
4. Record results cleanly so later agents can compare PyTorch/CUDA runs against the MLX findings.

## Read First

Before making any changes, read these files:
- [quantization_research_summary.md](./quantization_research_summary.md)
- [research/results_subset32768_skipfloat.tsv](./research/results_subset32768_skipfloat.tsv)
- [research/results.tsv](./research/results.tsv)
- [train_gpt.py](./train_gpt.py)
- [train_gpt_mlx.py](./train_gpt_mlx.py)

Then read the source MLX run scripts for the replication targets you are implementing. Those copied `research/<run_tag>/train_gpt_mlx.py` files are the source of truth for exact method behavior.

## Replication Targets

Replicate these runs first, in roughly this order:

| Priority | MLX run tag | Why replicate it |
| --- | --- | --- |
| 1 | `2026-03-28_mlx_quick_int6_group64_headint8_lzma_awqhead_calib128_val32768_skipfloat` | Best overall quantized result. |
| 2 | `2026-03-30_mlx_quick_int6_group64_tiedemb_fp16_lzma_rht_val32768_skipfloat` | Clean, general rotation baseline. |
| 3 | `2026-03-29_mlx_quick_int6_group64_headint8_lzma_awqheadonly_calib128_val32768_skipfloat` | Strong quality/size tradeoff. |
| 4 | `2026-03-28_mlx_quick_int5_group64_tiedemb_fp16_lzma_awq_calib128_val32768_skipfloat` | Strong compact AWQ branch. |
| 5 | `2026-04-02_mlx_quick_int5_group64_tiedemb_fp16_lzma_rht_signroundlite_calib128_rotseed2026_val32768_skipfloat` | Best SignRound-Lite result. |
| 6 | `2026-03-30_mlx_quick_int5_group64_headint6_lzma_awqheadonly_calib128_val32768_skipfloat` | Best sub-7MB branch. |
| 7 | `2026-04-02_mlx_quick_int4_group64_tiedemb_fp16_lzma_nf4_signroundlite_sparseresid001_calib128_val32768_skipfloat` | Best compact INT4-style branch. |
| 8 | `2026-04-02_mlx_quick_int4_group64_tiedemb_fp16_lzma_nf4_signroundlite_calib128_val32768_skipfloat` | Simpler NF4 + SignRound-Lite anchor. |
| 9 | `2026-03-29_mlx_quick_int4_group64_tiedemb_fp16_lzma_awq_calib256_val32768_skipfloat` | Compact INT4 + AWQ anchor. |
| 10 | `2026-03-29_mlx_quick_int4_group64_headint8_lzma_val32768_skipfloat` | Simplest extreme-compression anchor. |

Also replicate the quick baseline first:
- `2026-03-26_mlx_quickbaseline_int8_rtn_zlib_val32768_skipfloat`

## Core Principle

The PyTorch/CUDA work should be a faithful port of the MLX method stack, not a reinterpretation.

When in doubt:
- prefer matching MLX behavior over writing a prettier abstraction
- prefer matching env var names over inventing new knobs
- prefer matching logged metrics and log keys over renaming things
- prefer copied per-run PyTorch branches over editing the root `train_gpt.py` immediately

## Hard Rules

- Do not do new quantization research in this phase.
- Do not change dataset, tokenizer, evaluation math, or the 10 minute limit.
- Do not start reinvestment experiments here.
- Do not carry over architecture-specific fp16 keep-set hacks such as single-layer or single-tensor exceptions.
- Do not silently change artifact accounting by switching to a totally different serializer without noting it.
- Do not edit the MLX source runs. They are references.
- Do not rely on memory of how a method worked. Read the exact MLX run script for the run you are porting.

## Where To Work

Do not start by editing the root [train_gpt.py](./train_gpt.py) directly.

For each replication target:
1. Create `research/<run_tag>_cuda/`
2. Copy [train_gpt.py](./train_gpt.py) to `research/<run_tag>_cuda/train_gpt.py`
3. Create `research/<run_tag>_cuda/README.md`
4. Port only the method stack required for that run
5. Run and log that PyTorch branch

Only after several branches are working cleanly should you consider merging shared helpers back into a cleaner common CUDA branch.

## Required Parity Features

Before porting the top schemes, add the MLX quick-track infrastructure to the PyTorch code path:

- deterministic validation subset support via `VAL_SUBSET_SEQS`
- optional skipped final float validation via `SKIP_FINAL_FLOAT_VAL`
- structured per-run log file in `OUT_DIR/<RUN_ID>.txt`
- the existing final metric keys such as `final_int8_zlib_roundtrip_exact`
- logged artifact size using the existing parser-friendly key `artifact_bytes_int8_zlib`
- logged peak training memory using a CUDA equivalent of `train_peak_rss_bytes`
- quick-track run tagging via `RUN_ID` and `OUT_DIR`

Important:
- keep the same log keys if possible, even if they are slightly legacy-named
- later automation and existing TSV parsers already expect those strings

## Serialization Guidance

The current PyTorch baseline uses `torch.save(...)` plus `zlib`.
That is fine for the original baseline, but it is not automatically faithful to the MLX `lzma` frontier runs.

For the replicated frontier runs:
- keep the serialized quantized object as framework-agnostic as practical
- prefer raw Python containers and explicit tensor payloads over framework-specific checkpoint overhead
- keep artifact accounting consistent across the replicated PyTorch runs
- if you intentionally diverge from MLX serialization layout, state it explicitly in the per-run README

The goal is not perfect byte-for-byte identity with MLX.
The goal is a faithful method port where artifact size is still meaningful and comparable across the CUDA runs.

## Method Features You Will Need

Across the top replication set, PyTorch will need equivalents for:

- group-of-64 quantization for 2D tensors
- body bitwidth control via `QMAX_BODY`
- tied-head control via `QUANTIZE_TIED_HEAD` and `QMAX_HEAD`
- `lzma` serialization
- full AWQ
- head-only AWQ using include/exclude name patterns
- randomized Hadamard input-column rotation
- deterministic rotation seed handling via `ROTATION_SEED`
- scalar quantizers: at minimum `uniform` and `nf4`
- SignRound-Lite style reconstruction-guided rounding/clip search
- tiny sparse residual correction via `SPARSE_RESIDUAL_FRAC`

You do **not** need GPTQ-lite, FP4, low-rank residuals, or the discarded packbits paths in the first replication wave.

## Implementation Order

### Phase 1: Quick Baseline Parity

First port the MLX quick-track infrastructure onto PyTorch:
- `VAL_SUBSET_SEQS=32768`
- `SKIP_FINAL_FLOAT_VAL=1`
- run-tagged file logging
- artifact byte logging
- peak memory logging

Acceptance:
- the baseline PyTorch quick-track branch runs end-to-end
- the output log contains the same key metrics the MLX parser expects

### Phase 2: Simple Grouped + LZMA Branches

Port these first because they are simple and unblock later branches:
- INT4 group64 + `lzma` + INT8 head
- INT5 group64 + fp16 tied head
- INT6 group64 + fp16 tied head

Acceptance:
- artifact sizes are plausible
- evaluation runs after round-trip dequantization

### Phase 3: AWQ

Port:
- full AWQ
- tied-head AWQ
- head-only AWQ via include/exclude patterns

Use the same calibration subset semantics as MLX:
- deterministic prefix sampling
- matching sequence count
- matching calibration batch tokens

### Phase 4: Rotation

Port randomized Hadamard rotation exactly enough that:
- the same seed produces the same deterministic structural choice pattern
- rotation is fused into the quantization path, not treated as a separate architecture change

### Phase 5: NF4

Port the NF4 codebook branch next.
This unlocks the compact INT4 replication targets.

### Phase 6: SignRound-Lite And Sparse Residual

Port SignRound-Lite only after AWQ, grouped quantization, and NF4 are already working.

Then add:
- the reconstruction search grids
- the same calibration subset rules
- sparse residual support for the NF4 + SignRound-Lite branch

## Run Command

Use a copied branch, not the root file:

```bash
source .venv/bin/activate
export RUN_ID=<run_tag>_cuda
export OUT_DIR=research/<run_tag>_cuda
export VAL_SUBSET_SEQS=32768
export SKIP_FINAL_FLOAT_VAL=1
torchrun --standalone --nproc_per_node=8 research/<run_tag>_cuda/train_gpt.py \
  > research/<run_tag>_cuda/stdout.log \
  2> research/<run_tag>_cuda/time.log
```

If you are not on 8 GPUs, keep the `torchrun` invocation explicit and document the exact GPU count used.

## Results File

Log CUDA replication runs to:
- `research/results_cuda_replicate.tsv`

Suggested header:

```tsv
run_tag	source_mlx_run	val_bpb	artifact_bytes	time	train_peak_mem_bytes	status	description
```

Status guidance:
- `keep` if the run is a faithful successful replication worth carrying into reinvestment
- `maybe` if the method works but parity is incomplete or artifact accounting still differs materially
- `discard` if the port works but the scheme is no longer competitive on CUDA
- `crash` if it failed
- `invalid` if it violated the replication rules

## Acceptance Criteria

A replicated CUDA run is successful when:
- it is clearly traceable to one MLX source run
- it uses the same quick-track protocol
- it logs exact `val_bpb`
- it logs artifact bytes
- it logs peak training memory
- it stays within the challenge time budget
- it is close enough structurally that later agents can use it as the starting point for reinvestment

Do not require exact numerical identity with MLX.
Training backends differ.
What matters is faithful method implementation plus sensible relative behavior.

## What Success Looks Like

At the end of this phase, there should be:
- a PyTorch quick baseline with MLX-style quick-track logging
- 5 to 10 successful CUDA replications of the selected frontier runs
- a clean `research/results_cuda_replicate.tsv`
- a clear sense of which one or two replicated schemes should become the fixed bases for reinvestment
