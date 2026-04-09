# Parameter Golf Autoresearch Resume

## Introduction

This repository defines a model crafting challenge by OpenAI to train the best language model that fits in a 16MB artifact and trains in under 10 minutes.
More details about the challenge can be found in [README.md](./README.md).

This document details how you as an expert researcher agent are going to resume and continue research for this challenge autonomously, with minimal human input.

## Goal

Your job is to improve the final exact `val_bpb` for the Parameter Golf challenge using **quantization only**.

Work only on:
- weight quantization
- activation quantization
- quantization-aware training
- calibration for quantization
- packing / serialization / compression of quantized weights

Do **not** change other things unless a later follow-up experiment is explicitly testing capacity unlocked by a better quantization method.

The current training scripts use post-training quantization to INT8 (RTN) and compress with zlib. The model is then decompressed and run in higher precision for evaluation.

## Scope

The challenge provides two training scripts for [PyTorch/CUDA](./train_gpt.py) and [MLX/Apple Silicon](./train_gpt_mlx.py). These scripts are exactly as provided by OpenAI (unmodified) except for reducing `val_batch_size` and `train_batch_tokens` from 524,288 to 8092 in `train_gpt_mlx.py`.

All experimentation will be done on a MacBook (with Apple Silicon) for now.
Therefore, all your changes must only be made to `train_gpt_mlx.py`.
However, every change must have a clear PyTorch equivalent in `train_gpt.py`. We will migrate all positive changes on MLX to the PyTorch script and evaluate on 8xH100s.
But for now, all experimentation is on a MacBook.

In the current repo state, the root [train_gpt_mlx.py](./train_gpt_mlx.py) already includes resume-oriented helpers used by the active MLX loop:
- deterministic validation subset selection via `VAL_SUBSET_SEQS`
- optional skipped final float validation via `SKIP_FINAL_FLOAT_VAL`
- logged `artifact_bytes_int8_zlib`
- logged `train_peak_rss_bytes`

When resuming, copy the current root `train_gpt_mlx.py` into a new run directory unless you are explicitly reproducing an older run.

## Resume First

Before starting a new run, reconstruct state from files rather than memory:
- Read [research/results_subset32768_skipfloat.tsv](./research/results_subset32768_skipfloat.tsv) for the active quick-track local protocol.
- Read [research/results.tsv](./research/results.tsv) for the original full-validation track.
- Read the most relevant recent `research/<run_tag>/README.md` files only when you need hypothesis/change details beyond the tsv row.
- Do not redo old experiments unless you are intentionally reproducing a baseline for a new protocol.

Default resume assumption if the user gives no other guidance:
- Continue the quick-track MLX protocol that uses a fixed reduced validation subset.
- Keep comparisons within the same results file and protocol.
- Treat the baseline row in the chosen results file as the current reference point.

## Hard rules
These rules are strict:
- Do not change the 10 minute limit
- Do not change dataset, tokenizer, optimizer, LR schedule, batch size, sequence length, model size, or eval path in quantization-only runs
- Do not refactor unrelated code
- Make exactly one logical change per run. Do not combine multiple new ideas in one run.
- Do not report approximate metrics as the final score if an exact score is available
- Do not modify the original train_gpt_mlx.py
- In quantization-only runs, the only allowed changes are quantization-related.
- **Do not change the evaluation code!**
- Do not change how final metrics are computed or printed.
- Do not change random seed unless explicitly testing variance.
- Avoid hand-picked layer-specific or tensor-specific keep-sets unless the rule itself is algorithmic and architecture-agnostic.
- Structural or role-based exceptions are allowed when they transfer across architectures and sizes. For example, the tied embedding/output head is an acceptable general exception; "keep block 8 MLP proj in fp16" is not a preferred research direction now.

## Experimentation

Each experiment runs on a MacBook Pro with Apple Silicon (M4 Pro, 24GB unified memory). Each training run has a **maximum duration of 10 minutes (600 seconds). You must not change this limit!**

### Active Resume Protocol

The default local resume protocol is:
- `VAL_SUBSET_SEQS=32768`
- `SKIP_FINAL_FLOAT_VAL=1`
- results logged to [research/results_subset32768_skipfloat.tsv](./research/results_subset32768_skipfloat.tsv)

Important notes:
- The validation subset must be identical across quick-track runs. In the current script, `VAL_SUBSET_SEQS` selects a deterministic fixed prefix, so runs are comparable as long as the value stays the same.
- Because this protocol changes the evaluation set and skips the final float-model validation pass, its absolute `val_bpb` values are not directly comparable to the original full-validation baseline.
- For quick-track runs, use the baseline row in [research/results_subset32768_skipfloat.tsv](./research/results_subset32768_skipfloat.tsv) as the soft memory ceiling via its `train_peak_rss_bytes` value. Slight overage is acceptable, but future runs should not materially exceed it.
- Be careful interpreting nominal bitwidth changes: in the current experimental code paths, quantized values are often still stored in `int8` arrays unless a packing experiment explicitly changes the storage format. That means grouped scales and serialization choices can move artifact bytes more than changing `qmax` alone.
- Be careful interpreting group-size sweeps: compressed artifact bytes are not guaranteed to decrease monotonically with larger groups because the saved scale bytes interact with zlib and pickle overhead in non-obvious ways. Treat group size as an empirical axis, not a purely arithmetic one.
- The simple tier-2 extension of weighting clip search by diagonal activation energy was not enough. When resuming calibration-based PTQ, prefer richer activation-aware objectives such as sampled output error, covariance-aware scoring, AWQ/GPTQ-style criteria, or true sequential error compensation.
- Lossless bit-packing under `zlib` can backfire badly. In particular, packing INT5 values before `zlib` made the compressed artifact larger than the original `int8`-stored version because the byte-level redundancy that `zlib` exploited was destroyed.
- Codec choice itself is a real frontier. Offline recompression of the strongest grouped raw pickles showed that `lzma` can beat `zlib` materially on artifact size (roughly 0.6 to 1.1 MB on current frontier runs), while `gzip` is basically identical to `zlib` and `bz2` is only a modest improvement.
- For pure serialization experiments, full retraining can obscure the true codec effect because the quantized tensors themselves are unchanged. When resuming codec work, prefer confirming codec size gains by recompressing the same raw pickle from an existing run before deciding whether a full train-and-eval rerun is worth spending.
- Breadth-first method comparison is now the preferred strategy. The current reusable breadth script branch supports three families via env vars in copied per-run scripts:
  - `QUANT_METHOD=rtn|gptq`
  - `INPUT_ROTATION=none|hadamard`
  - `SCALAR_QUANTIZER=uniform|normal_quantile`
  - `GPTQ_*` calibration knobs for a block-diagonal group-of-64 Hessian proxy
- Under the current local protocol, evaluation still dequantizes weights back to float and runs the normal model forward. That means full low-bit activation or KV-cache methods are **not** apples-to-apples here unless you intentionally change the forward path. For local breadth work, treat activation-aware methods mostly as calibration for weight quantization, and treat the current `normal_quantile` rotated path as a **TurboQuant-style weight-only proxy**, not a full TurboQuant reproduction.

### Experiment Types
There are 2 experiment classes:

**A. Quantization Only**
Keep training and model settings fixed. Change only quantization.
Examples:
- INT8 -> INT6 -> INT4 RTN
- per-tensor/per-row/per-group scales
- GPTQ/AWQ style calibration
- Activation quantization
- QAT
- Codebooks
- Entropy coding
- Better bit packing

The aim of these experiments is to reduce artifact size with small/minimal quality loss.
A quantization-only run does not need to beat baseline directly.
Its job may be to unlock a better follow-up run with more parameters.

**B. Reinvestment**
Only allowed after a quantization method looks promising.

Goal: use saved bytes or faster low-bit compute to test a larger model or more training.

Examples:
- more layers
- wider model
- larger batch
- more steps

Do not start here. First prove the quantization method helps.

### Setup
To set up a new experiment:
1. Read the relevant results file first so you do not repeat completed work:
    - quick-track: `research/results_subset32768_skipfloat.tsv`
    - full-track: `research/results.tsv`
2. Define a run tag/name and create a new folder `research/<run_tag>`
3. Read the in-scope files:
    - `README.md` for repository context
    - `train_gpt_mlx.py` that defines train + eval on MLX, and is the file you will modify
    - `train_gpt.py` to compare to the MLX version and ensure equivalent changes exist for PyTorch.
4. Copy `train_gpt_mlx.py` to `research/<run_tag>/train_gpt_mlx.py`
5. Create `research/<run_tag>/README.md`
6. Verify data exists: `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`. If not, tell human to check.
7. Confirm and go.

### Run Command
To launch a run, prefer the quick-track command unless the user explicitly wants full validation again.

Quick-track default resume command:
```bash
source .venv/bin/activate
export RUN_ID=<run_tag>
export OUT_DIR=research/<run_tag>
export VAL_SUBSET_SEQS=32768
export SKIP_FINAL_FLOAT_VAL=1
/usr/bin/time -p python3 research/<run_tag>/train_gpt_mlx.py \
  > research/<run_tag>/stdout.log \
  2> research/<run_tag>/time.log
```

Full-track command if you intentionally want the original protocol:
```bash
source .venv/bin/activate
export RUN_ID=<run_tag>
export OUT_DIR=research/<run_tag>
/usr/bin/time -p python3 research/<run_tag>/train_gpt_mlx.py \
  > research/<run_tag>/stdout.log \
  2> research/<run_tag>/time.log
```

Notes:
- The script itself writes the structured experiment log to `research/<run_tag>/<run_tag>.txt`.
- `stdout.log` may stay empty because the script logs directly to its own run-tagged file. That is normal.
- Redirect everything - do NOT use tee or let output flood your context.
- For paper/source review from the local CLI, use:
```bash
source .venv/bin/activate
hf papers search "activation-aware weight quantization"
hf papers read 2306.00978
```

### Queued Batch Command
When scheduling multiple unattended quick-track runs, prefer the queue runner plus a detached `tmux` session so the batch survives session end:

```bash
source .venv/bin/activate
tmux new-session -d -s awq_queue_20260328 \
  'cd /Users/jeebho01/parameter-golf && \
   python3 -u research/run_mlx_experiment_queue.py \
     research/queue_specs/2026-03-28_awq_queue.json \
     > research/queue_specs/2026-03-28_awq_queue.tmux.log 2>&1'
```

Queue notes:
- The queue spec lives in `research/queue_specs/*.json`.
- The runner upserts parsed metrics into the results file named in the spec after each finished run.
- Track the detached batch with `tmux ls`, inspect live output with `tmux capture-pane -pt <session>:0`, and stop it with `tmux kill-session -t <session>`.
- To resume without memory, inspect the queue specs directly rather than assuming which families are staged. Recent specs in this repo cover breadth-first families such as GPTQ-lite, randomized-Hadamard rotation, sparse residual correction, and low-rank residual reconstruction.
- You can chain unattended batches by launching a second detached `tmux` session that waits for the first session to disappear, then starts the next queue. This keeps the machine busy across session boundaries without manual intervention.

#### Baseline
Baseline results when running `train_gpt_mlx.py` with quick-track evaluation (i.e. VAL_SUBSET_SEQS=32768, SKIP_FINAL_FLOAT_VAL=1):
```
final_int8_zlib_roundtrip val_loss:3.2115 val_bpb:1.9086 eval_time:396052ms
final_int8_zlib_roundtrip_exact val_loss:3.21146495 val_bpb:1.90859022
```

The score to beat is the exact metric 1.90859022.
Experiments **must not** take longer than the baseline.
Experiments must no exceed the (soft) memory ceiling set by the baseline - to ensure fair comparison.

Baseline results when running `train_gpt_mlx.py` with full evaluation (default):
```
final_int8_zlib_roundtrip val_loss:3.2003 val_bpb:1.8954 eval_time:707181ms
final_int8_zlib_roundtrip_exact val_loss:3.20025553 val_bpb:1.89537158
python3 train_gpt_mlx.py  236.01s user 69.98s system 15% cpu 33:47.19 total
```

### Logging Results
Use the results file that matches the active protocol:
- full-track: `research/results.tsv`
- quick-track subset-32768 skip-float: `research/results_subset32768_skipfloat.tsv`

Full-track header:
```tsv
run_tag val_bpb artifact_bytes  time  status  description
```

Quick-track header:
```tsv
run_tag val_bpb artifact_bytes time train_peak_rss_bytes status description
```

Fields:
1. `run_tag`
2. final exact `val_bpb` (`0.000000` if crash)
3. final artifact size in bytes (`0.000000` if crash). artifact_bytes = code bytes + compressed model bytes (not in-memory model).
4. time taken for run
5. `train_peak_rss_bytes` for protocols that record memory
6. status: `keep`, `maybe`, `discard`, `crash`, `invalid`
7. short description of the experiment

### Collecting Results

Prefer extracting exact metrics from the run-tagged log file plus `time.log`.

For a run at `research/<run_tag>/`, collect:
- `final_int8_zlib_roundtrip_exact ... val_bpb:...` from `research/<run_tag>/<run_tag>.txt`
- `artifact_bytes_int8_zlib:...` from `research/<run_tag>/<run_tag>.txt`
- `train_peak_rss_bytes:...` from `research/<run_tag>/<run_tag>.txt` when available
- wallclock from `research/<run_tag>/time.log`

Useful command pattern:
```bash
rg -n "final_int8_zlib_roundtrip_exact|artifact_bytes_int8_zlib|train_peak_rss_bytes|skipping_final_float_val|stopping_early" \
  research/<run_tag>/<run_tag>.txt
cat research/<run_tag>/time.log
```

A quantization-only run can still be `keep` or `maybe` even if `val_bpb` is worse than baseline, if it saves enough bytes to justify a later reinvestment run:
- `keep` if val_bpb improves
- `keep` if val_bpb is slightly worse but byte savings are substantial
- `maybe` if byte savings are moderate and quality drop is small
- `discard` if val_bpb gets worse and byte savings are small
- `discard` if eval gets much more complex or unstable for little gain

### Experiment Loop

For every run:
1. Create a new run directory `research/<run_tag>`
2. Copy `train_gpt_mlx.py` to `research/<run_tag>/train_gpt_mlx.py`
3. Create a README in `research/<run_tag>`.
4. Make one change only to `research/<run_tag>/train_gpt_mlx.py`. Do not modify the original python file, which must remain as is for baselines.
5. Run the script with the command that matches the current protocol and save logs.
6. Extract exact `val_bpb`, `artifact_bytes`, runtime, and `train_peak_rss_bytes` if present.
7. Update the appropriate results file and the run README.
8. Move to next run.

Once a run starts, do not edit that run's code. Any code change requires a new run_tag.

It is very important to only change one thing at a time in each experiment, and run thorough ablations e.g. when testing a new quantization scheme, do not also increase the number of layers.
1. First test the new quantization scheme (this may increase val_bpb due to increased compression)
2. Then increase #parameters in a new run (hopefully, this will decrease val_bpb)

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~10 minutes then you can run approx 6/hour, for a total of about 50 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

#### Per-run README
Each research/<run_tag>/README.md must contain:
- hypothesis
- exact change made
- why this is in scope
- expected effect on val_bpb
- expected effect on artifact bytes
- final result
- short conclusion
Be concise. Use bullet points.

#### Crashes
If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

### Search order

**Tier 1**
- INT4/6
- per-tensor vs per-row vs per-group scales
- better bit packing
- better serialization

**Tier 2**
- calibration-based PTQ e.g. GPTQ, AWQ.
- activation quantization
- mixed precision for outliers
- scale quantization
- entropy coding of quantized weights or indices

**Tier 3**
- NF4 style codebooks
- QAT
- fake quant during training
- learned quantizer scales
- FP4/MXFP4/NVFP4 (if native hardware support)

**Tier 4**
- INT3 / INT2
- Hadamard / incoherence methods e.g. QuaRot / SpinQuant / QuiP#
- BitNet-style ideas

This list is not exhaustive. You may conduct research and read papers from online to find interesting and novel ideas. 

You can use `hf papers` CLI tool to search, discover, and read academic papers. For example, try:
```bash
source .venv/bin/activate
hf papers search "llm quantization"
hf papers read 2310.19102
```
This is very useful to search for new research directions, but also to confirm that your understanding of methods is correct. It is always best to check the source material before implementing a specific method.

### What good research looks like
Good research is:
- small diffs
- one variable changed
- clear logs
- exact score recorded
- failures recorded
- no repeated mistakes
- easy to port to PyTorch later
