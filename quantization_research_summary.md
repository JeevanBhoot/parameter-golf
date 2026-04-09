# Quantization Research Summary

## Scope

This repository ran a broad MLX-first quantization search for the Parameter Golf challenge, then narrowed to a small set of strong, generalizable schemes worth porting to CUDA/PyTorch.

The exact run ledger lives in:
- [research/results.tsv](./research/results.tsv) for the original full-validation kickoff
- [research/results_subset32768_skipfloat.tsv](./research/results_subset32768_skipfloat.tsv) for the main quick-track protocol

As of April 9, 2026, the logged search contains:
- 2 full-track MLX runs
- 122 quick-track MLX runs
- 124 total logged MLX runs

Most conclusions below come from the quick-track protocol:
- `VAL_SUBSET_SEQS=32768`
- `SKIP_FINAL_FLOAT_VAL=1`
- quick baseline: `val_bpb=1.90859022`, `artifact_bytes=12972589`, `train_peak_rss_bytes=1534263296`

## Executive Summary

- The strongest overall family was grouped low-bit quantization plus better serialization. `lzma` mattered a lot.
- General, transferable wins came from four ideas: grouped INT4/5/6, AWQ-style activation-aware scaling, randomized Hadamard rotation, and better 4-bit codebooks such as NF4/AF4.
- The best reconstruction-style addition was `signroundlite`, a SignRound-inspired per-group rounding and clip search.
- A small sparse residual on top of NF4 + SignRound-Lite was one of the best compact INT4 results.
- GPTQ-lite underperformed AWQ and rotation on this setup.
- Packbits-style manual bit-packing was usually counterproductive once compression was applied.
- FP4 was not competitive here.
- Early layer-specific fp16 keep-sets gave good local numbers, but they are not the right long-term direction because they do not transfer cleanly across architectures and scales.

## What Was Tested

The complete row-by-row record is in the TSV files. The table below summarizes every research thread that was explored. Family counts overlap because many later runs combined multiple ideas.

| Thread | What was tested | Best representative | Result | Conclusion |
| --- | --- | --- | --- | --- |
| Full-track kickoff | Original MLX baseline and one grouped INT8 rerun | `2026-03-26_mlx_baseline_int8_rtn_zlib` | `1.89537158`, `13096769` bytes | Established the original full-validation reference, then the project moved to the quicker subset protocol. |
| Quick baseline | Quick baseline with fixed subset and skipped final float eval | `2026-03-26_mlx_quickbaseline_int8_rtn_zlib_val32768_skipfloat` | `1.90859022`, `12972589` bytes | Main local reference for all later comparisons. |
| Early architecture-specific fp16 keep-sets | `tok_emb`, `allcv`, `mlpproj`, `mlpfc`, CQ-style keep-sets | `2026-03-26_mlx_quick_b8_mlpproj_fp16_val32768_skipfloat` | `1.89782535`, `13806627` bytes | Strong local quality, but not recommended for GPU carry-forward because these are too architecture-specific. |
| Plain grouped RTN bitwidth sweeps | INT6, INT5, INT4, group-of-64, tied-head precision sweeps | `2026-03-27_mlx_quick_int6_group64_tiedemb_fp16_val32768_skipfloat` | `1.90376429`, `10045417` bytes | Grouped INT6/5/4 was the base frontier that unlocked the rest of the search. |
| Manual bit-packing | `packbits` variants for INT6 and INT5 | `2026-03-27_mlx_quick_int6_rtn_packbits_val32768_skipfloat` | `1.91314486`, `10531333` bytes | Packing hurt compressed size; not worth carrying forward. |
| Better serialization | `lzma` instead of `zlib` on the strongest grouped branches | `2026-03-29_mlx_quick_int4_group64_tiedemb_fp16_lzma_val32768_skipfloat` | `1.94251809`, `4997929` bytes | Clear win. Codec choice was a real research axis, not bookkeeping. |
| Head precision and tied-head sweeps | fp16 tied head vs INT8/7/6 head | `2026-03-30_mlx_quick_int5_group64_headint6_lzma_awqheadonly_calib128_val32768_skipfloat` | `1.90805578`, `6750137` bytes | A structural head exception is general and useful. Head bitwidth became one of the most effective compactness knobs. |
| AWQ / activation-aware scaling | Full-body AWQ and tied-head AWQ | `2026-03-28_mlx_quick_int6_group64_headint8_lzma_awqhead_calib128_val32768_skipfloat` | `1.89650511`, `9525023` bytes | One of the strongest families overall. AWQ consistently helped. |
| Head-only AWQ | AWQ only on the tied embedding/output head | `2026-03-29_mlx_quick_int6_group64_headint8_lzma_awqheadonly_calib128_val32768_skipfloat` | `1.89921697`, `8862485` bytes | Very strong tradeoff. A good middle ground between full AWQ and simpler grouped RTN. |
| Randomized Hadamard rotation | RHT/QuaRot-style input-column rotation | `2026-03-30_mlx_quick_int6_group64_tiedemb_fp16_lzma_rht_val32768_skipfloat` | `1.89738264`, `9399930` bytes | One of the cleanest general wins. Worked especially well at INT6 and in some codebook families. |
| Rotation seed sweep | SpinQuant-inspired deterministic seed sweep | `2026-04-02_mlx_quick_int5_group64_tiedemb_fp16_lzma_rht_signroundlite_calib128_rotseed2026_val32768_skipfloat` | `1.90656680`, `7369643` bytes | Seed mattered for some families, but only modestly. Important as a refinement, not a new family. |
| GPTQ-lite | Block-diagonal Hessian proxy with sequential compensation | `2026-03-30_mlx_quick_int6_group64_tiedemb_fp16_lzma_gptqlite_calib128_val32768_skipfloat` | `1.90385399`, `9162008` bytes | Respectable, but consistently weaker than AWQ or rotation on this setup. |
| Normal-quantile proxy | TurboQuant-style weight-only proxy via rotated normal quantiles | `2026-03-30_mlx_quick_int6_group64_tiedemb_fp16_lzma_rht_normalq_val32768_skipfloat` | `1.90064065`, `11079622` bytes | Interesting, but not competitive enough to keep pursuing in this MLX setup. |
| Sparse residual correction | Keep top residual weights after quantization | `2026-03-30_mlx_quick_int5_group64_headint8_lzma_sparseresid001_val32768_skipfloat` | `1.90897303`, `6986041` bytes | Modest gains in a few spots, especially as a tiny refinement on already-good compact schemes. |
| Low-rank residual correction | Rank-1 residual and activation-shaped low-rank correction | `2026-03-30_mlx_quick_int5_group64_tiedemb_fp16_lzma_lowrankr1_val32768_skipfloat` | `1.90809321`, `7300404` bytes | Not bad, but weaker than the best AWQ/rotation/SignRound combinations. |
| NF4 codebooks | NF4 with and without rotation, plus NF4 combinations | `2026-04-02_mlx_quick_int4_group64_tiedemb_fp16_lzma_nf4_signroundlite_awq_calib128_val32768_skipfloat` | `1.91179029`, `7105731` bytes | NF4 became genuinely strong once combined with better calibration or reconstruction. |
| FP4 codebooks | FP4 and rotated FP4 | `2026-04-02_mlx_quick_int4_group64_tiedemb_fp16_lzma_fp4_val32768_skipfloat` | `1.95110242`, `6563991` bytes | Not competitive here. |
| AF4 codebooks | AF4 and rotated AF4, then SignRound follow-ups | `2026-04-02_mlx_quick_int4_group64_tiedemb_fp16_lzma_rht_af4_signroundlite_calib128_val32768_skipfloat` | `1.91070634`, `7038923` bytes | AF4 became interesting only with rotation and reconstruction, but still was not the main frontier winner. |
| SignRound-Lite | Reconstruction-guided rounding and clip search | `2026-04-02_mlx_quick_int5_group64_tiedemb_fp16_lzma_rht_signroundlite_calib128_rotseed2026_val32768_skipfloat` | `1.90656680`, `7369643` bytes | Strong new family. Especially good on rotated INT5 and some codebook branches. |
| SignRound-Lite depth combos | AWQ + SignRound-Lite, NF4 + SignRound-Lite + sparse residual, AF4 + SignRound-Lite | `2026-04-02_mlx_quick_int4_group64_tiedemb_fp16_lzma_nf4_signroundlite_sparseresid001_calib128_val32768_skipfloat` | `1.91295851`, `6453579` bytes | The best late-stage compact INT4 results came from these combination runs. |

## Main Findings By Theme

### 1. Serialization Was A First-Class Lever

The move from the original `zlib` path to `lzma` changed the frontier materially. This was one of the most reliable byte-saving improvements in the whole project.

### 2. AWQ Was The Most Consistently Useful Calibration Method

AWQ improved both INT6 and INT5 families, and it also worked well in structural forms such as head-only AWQ. The best overall run used full AWQ on the INT6 body and the tied head.

### 3. Randomized Hadamard Rotation Was A Clean, General Win

RHT was especially strong on INT6 and helped several codebook branches. It was one of the most attractive ideas for PyTorch/CUDA carry-over because it is general, architecture-agnostic, and not tied to specific layers.

### 4. GPTQ-Lite Did Not Win This Local Contest

The second-order proxy worked, but it did not beat AWQ or the best rotation-based methods on the same byte range. It should not be a first replication target.

### 5. NF4 Was Worth It Once Paired With Better Rounding

Plain NF4 was decent. NF4 plus SignRound-Lite was clearly better. NF4 plus SignRound-Lite plus a tiny sparse residual became one of the strongest compact results in the entire search.

### 6. SignRound-Lite Was The Best New Late-Stage Idea

The strongest late additions came from reconstruction-guided rounding search. The best result in that family used rotated INT5 with seed `2026`, and the best compact INT4 variant used NF4 plus SignRound-Lite plus a `0.1%` sparse residual.

### 7. Some Early Wins Were Intentionally Not Carried Forward

Several layer-specific fp16 keep-set runs were good numerically, but they do not match the long-term goal of a general compression method that can be reused across sizes and architectures. They are important history, not recommended replication targets.

## Recommended CUDA/PyTorch Replication Set

This is the shortlist I would repeat on GPU. It is not a pure sort by `val_bpb`; it is a practical Pareto-style shortlist that spans the most useful byte regimes and the most credible method families.

Two of the runs below still carry `review` in the TSV, but their exact metrics are already logged and they are worth replicating.

| Regime | Run tag | Scheme | `val_bpb` | Artifact bytes | Why it made the cut |
| --- | --- | --- | ---: | ---: | --- |
| Quality-first | `2026-03-28_mlx_quick_int6_group64_headint8_lzma_awqhead_calib128_val32768_skipfloat` | INT6 group64 + INT8 head + full AWQ including tied head | `1.89650511` | `9525023` | Best overall quantized result in the quick-track search. |
| Quality-first | `2026-03-30_mlx_quick_int6_group64_tiedemb_fp16_lzma_rht_val32768_skipfloat` | INT6 group64 + randomized Hadamard rotation + fp16 tied head | `1.89738264` | `9399930` | Clean, simple, very general rotation-based baseline. |
| Balanced | `2026-03-29_mlx_quick_int6_group64_headint8_lzma_awqheadonly_calib128_val32768_skipfloat` | INT6 group64 + INT8 head + head-only AWQ | `1.89921697` | `8862485` | Excellent size/quality balance with low implementation complexity. |
| Balanced | `2026-03-28_mlx_quick_int5_group64_tiedemb_fp16_lzma_awq_calib128_val32768_skipfloat` | INT5 group64 + fp16 tied head + AWQ | `1.90306011` | `7871147` | Stronger compact branch than plain INT5 RTN, and a solid AWQ reference. |
| Balanced | `2026-04-02_mlx_quick_int5_group64_tiedemb_fp16_lzma_rht_signroundlite_calib128_rotseed2026_val32768_skipfloat` | INT5 group64 + RHT + SignRound-Lite + seed 2026 | `1.90656680` | `7369643` | Best SignRound-Lite result and the best demonstration that reconstruction search is real. |
| Compact | `2026-03-30_mlx_quick_int5_group64_headint6_lzma_awqheadonly_calib128_val32768_skipfloat` | INT5 group64 + INT6 head + head-only AWQ | `1.90805578` | `6750137` | One of the best sub-7MB results. Very attractive for reinvestment. |
| Compact | `2026-04-02_mlx_quick_int4_group64_tiedemb_fp16_lzma_nf4_signroundlite_sparseresid001_calib128_val32768_skipfloat` | INT4 group64 + NF4 + SignRound-Lite + `0.1%` sparse residual | `1.91295851` | `6453579` | Best compact INT4-style result with a generalizable method stack. |
| Compact | `2026-04-02_mlx_quick_int4_group64_tiedemb_fp16_lzma_nf4_signroundlite_calib128_val32768_skipfloat` | INT4 group64 + NF4 + SignRound-Lite | `1.91733550` | `6365991` | Simpler sibling of the previous run. Important if sparse residuals are annoying to port first. |
| Ultra-compact | `2026-03-29_mlx_quick_int4_group64_tiedemb_fp16_lzma_awq_calib256_val32768_skipfloat` | INT4 group64 + fp16 tied head + AWQ with 256 calibration sequences | `1.92713597` | `5901309` | Good compact AWQ-only anchor in the sub-6MB range. |
| Ultra-compact | `2026-03-29_mlx_quick_int4_group64_headint8_lzma_val32768_skipfloat` | Plain INT4 group64 + INT8 head + lzma | `1.94109118` | `4736241` | Simplest extreme-compression anchor; useful sanity baseline for GPU replication. |

## Runs I Would Not Prioritize For GPU Porting

- Architecture-specific fp16 keep-sets such as `b8_mlpproj_fp16`, `tokemb_fp16`, `allcv_fp16`, and similar runs.
- Manual `packbits` branches.
- GPTQ-lite branches.
- The rotated normal-quantile proxy branch.
- FP4 branches.
- Most plain sparse-residual and low-rank correction branches that were not paired with stronger codebook/reconstruction methods.

## Suggested GPU Replication Order

1. Quick baseline parity on PyTorch/CUDA.
2. Plain grouped RTN + `lzma` + head precision support.
3. AWQ and head-only AWQ branches.
4. RHT rotation.
5. NF4 codebooks.
6. SignRound-Lite.
7. Sparse residual on top of NF4 + SignRound-Lite.

## Bottom Line

If you only replicate a few things, make them these:
- INT6 + full AWQ + INT8 head
- INT6 + RHT
- INT5 + AWQ
- INT5 + RHT + SignRound-Lite
- INT4 + NF4 + SignRound-Lite, with and without the tiny sparse residual

Those cover the strongest quality-first, balanced, and compact regions of the frontier while staying aligned with the user's preference for general methods rather than hand-picked layer exceptions.
