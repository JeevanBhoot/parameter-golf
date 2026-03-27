# Parameter Golf Autoresearch

## Introduction

This repository defines a model crafting challenge by OpenAI to train the best language model that fits in a 16MB artifact and trains in under 10 minutes.
More details about the challenge can be found in [README.md](./README.md).

This document details how you as an expert researcher agent are going to conduct research for this challenge autonomously, with minimal human input.

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

## Experimentation

Each experiment runs on a MacBook Pro with Apple Silicon (M4 Pro, 24GB unified memory). Each training run has a **maximum duration of 10 minutes (600 seconds). You must not change this limit!**

The first run: Your very first run should always be to establish the baseline, so you will run the training script as is.

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
1. Define a run tag/name and create a new folder `research/<run_tag>`
2. Read the in-scope files:
    - `README.md` for repository context
    - `train_gpt_mlx.py` that defines train + eval on MLX, and is the file you will modify
    - `train_gpt.py` to compare to the MLX version and ensure equivalent changes exist for PyTorch.
3. Copy `train_gpt_mlx.py` to `research/<run_tag>/train_gpt_mlx.py`
4. Create `research/<run_tag>/README.md`
4. Verify data exists: `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`. If not, tell human to check.
5. Confirm and go.

### Run Command
To launch a run:
```bash
# Activate virtual environment
source .venv/bin/activate
# Launch run
time python3 research/<run_tag>/train_gpt_mlx.py > research/<run_tag>/run.log 2>&1
```
Redirect everything - do NOT use tee or let output flood your context

#### Baseline
Baseline results when running `train_gpt_mlx.py` without any changes:
```
final_int8_zlib_roundtrip val_loss:3.2003 val_bpb:1.8954 eval_time:707181ms
final_int8_zlib_roundtrip_exact val_loss:3.20025553 val_bpb:1.89537158
python3 train_gpt_mlx.py  236.01s user 69.98s system 15% cpu 33:47.19 total
```

The score to beat is the exact metric 1.89537158.
Experiments **must not** take longer than the baseline.

### Logging Results
Store results in `research/results.tsv`.

Header:
```tsv
run_tag val_bpb artifact_bytes  time  status  description
```

Fields:
1. `run_tag`
2. final exact `val_bpb` (`0.000000` if crash)
3. final artifact size in bytes (`0.000000` if crash). artifact_bytes = code bytes + compressed model bytes (not in-memory model).
4. time taken for run
5. status: `keep`, `maybe`, `discard`, `crash`, `invalid`
6. short description of the experiment

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
5. Run the script and save logs.
6. Extract val_bpb and artifact size.
7. Update `research/results.tsv` and run README.
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

### What good research looks like
Good research is:
- small diffs
- one variable changed
- clear logs
- exact score recorded
- failures recorded
- no repeated mistakes
- easy to port to PyTorch later