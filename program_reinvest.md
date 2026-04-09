# Parameter Golf Reinvestment Program

## Mission

Your job is to use a **fixed, already-replicated quantization scheme** to buy a better model.

This phase begins only after CUDA/PyTorch replication is working.
You are no longer searching for better quantization.
You are searching for the best use of the savings unlocked by an already-good quantization scheme.

Typical savings to reinvest:
- artifact bytes
- training memory
- training throughput

The goal is simple:
- keep the submission under the challenge limits
- spend the savings where they improve exact `val_bpb`

## Read First

Before starting reinvestment, read:
- [quantization_research_summary.md](./quantization_research_summary.md)
- [program_replicate.md](./program_replicate.md)
- `research/results_cuda_replicate.tsv` once it exists
- the replicated CUDA run directory for the chosen base scheme
- [train_gpt.py](./train_gpt.py)

## Prerequisite

Pick one replicated CUDA quantization scheme and freeze it for a reinvestment branch.

Good starting base choices:
- quality-first base: INT6 group64 + INT8 head + full AWQ
- balanced base: INT5 group64 + AWQ or INT5 group64 + RHT + SignRound-Lite
- capacity-first base: INT4 group64 + NF4 + SignRound-Lite, optionally with tiny sparse residual

Within a reinvestment branch:
- keep the quantization method fixed
- only change architecture or training-budget usage
- do not simultaneously start changing the quantization algorithm again

## Hard Rules

- Stay under the 10 minute limit.
- Stay under the 16MB artifact limit.
- Do not change dataset, tokenizer, or evaluation math.
- Keep the chosen quantization scheme fixed for an entire reinvestment branch.
- Make one logical change per run.
- Do not hide extra compute inside evaluation.
- Do not quietly drop the quick-track protocol when comparing reinvestment runs locally.

## First Step: Build A Budget Ledger

Before proposing reinvestment experiments, measure the chosen CUDA base scheme:
- artifact bytes
- peak training memory
- wallclock time
- tokens per second
- exact `val_bpb`

Then compute:
- `byte_headroom = 16_000_000 - artifact_bytes`
- `time_headroom = 600 - wallclock_seconds`
- `memory_headroom = hardware_limit - peak_training_memory`

Use real measured headroom, not guesses.

## Results File

Log reinvestment runs to:
- `research/results_reinvest.tsv`

Suggested header:

```tsv
run_tag	base_quant_run	val_bpb	artifact_bytes	time	train_peak_mem_bytes	status	description
```

## Search Strategy

Reinvestment should be breadth-first at first, then depth.
Do not jump straight to exotic architecture changes before you have mapped the obvious scaling surface.

## Tier 1: Simple Capacity Reinvestment

These are the first things to try.
They are easy to reason about and easy to compare.

### 1. More Layers

Try small layer increases first:
- `NUM_LAYERS +1`
- `NUM_LAYERS +2`
- `NUM_LAYERS +3`

This is often the cleanest first use of byte savings.

### 2. Wider Model

Increase `MODEL_DIM` carefully.
When widening:
- keep head geometry valid
- keep group sizes and quantization shapes sensible
- watch training memory closely

### 3. Bigger MLP

Try `MLP_MULT` increases after the layer sweep.
This is a good second lever once depth has been mapped.

### 4. More Training Compute

If the quantized CUDA path is faster:
- increase `ITERATIONS`
- increase batch size
- adjust warmdown only if needed

This is especially relevant if the chosen quant scheme reduces training memory or improves throughput enough to spend more compute inside the same 600 seconds.

## Tier 2: Conservative Architecture Improvements

Once the simple scaling surface is mapped, test higher-upside but still transformer-adjacent ideas.

### 1. Attention Residual Variants

Examples:
- a gated residual branch on the attention output
- an extra lightweight residual path around attention
- learned residual scaling for attention and MLP outputs

These can improve optimization without blowing up bytes too aggressively.

### 2. Gated MLP Variants

Examples:
- SwiGLU-style or gated feedforward replacements
- small gated expansion changes

These often improve parameter efficiency, which makes them good reinvestment targets.

### 3. Parallel Or Hybrid Residual Layouts

Examples:
- parallel attention + MLP residual updates
- shared pre-norm with lightweight extra residual branches

Keep these changes small and interpretable.

## Tier 3: Higher-Risk, Higher-Upside Reinvestment

Only go here after Tier 1 and Tier 2 have been mapped.

### 1. Attention Residual Memory Ideas

Examples:
- a small persistent residual memory branch
- cross-block residual accumulation mechanisms
- lightweight learned state carried across layers

### 2. Delta / State-Space Hybrids

Examples:
- gated delta-rule inspired blocks
- Gated DeltaNet-style hybrid branches
- replacing every Nth transformer block with a lightweight recurrent or delta-style mixer

Do not treat this as a vague brainstorm.
If you test this class, define one very specific architecture change per run and compare it against the same frozen quantization base.

### 3. Mixed Reinvestment Patterns

Examples:
- slightly more layers plus slightly more steps
- slightly wider model plus tighter head quantization that is already fixed inside the chosen scheme

Do not combine changes until the single-axis sweeps are mapped first.

## Recommended Reinvestment Order

For each chosen base quant scheme:

1. Layer sweep
2. Width sweep
3. MLP sweep
4. Training-compute sweep
5. One conservative residual/gating idea
6. One higher-risk hybrid idea such as a delta-style branch

Only after the first five are mapped should you start stacking changes.

## Suggested Base Schemes

You do not need to run reinvestment on every replicated quant scheme.
Pick 2 or 3 bases that cover different operating points.

Recommended:
- quality-first base: INT6 + full AWQ + INT8 head
- balanced base: INT5 + RHT + SignRound-Lite or INT5 + full AWQ
- compact base: INT4 + NF4 + SignRound-Lite + sparse residual

This gives you three different tradeoff regimes without exploding the search space.

## Per-Run Process

For each reinvestment run:

1. Start from one fixed CUDA quantization base.
2. Create a fresh `research/<run_tag>/train_gpt.py`.
3. Make exactly one logical reinvestment change.
4. Run it.
5. Record exact `val_bpb`, artifact bytes, wallclock time, and peak training memory.
6. Update `research/results_reinvest.tsv`.
7. Write a short README with the hypothesis and outcome.

## Evaluation Guidance

Keep local comparisons on the same quick-track protocol first:
- `VAL_SUBSET_SEQS=32768`
- `SKIP_FINAL_FLOAT_VAL=1`

Once a reinvestment run is clearly promising, it can later be promoted to fuller validation.
Do not mix quick-track and full-track rows in the same comparison table.

## Status Guidance

- `keep` if a reinvestment run improves exact `val_bpb` over its own fixed quant base while respecting artifact and time limits
- `maybe` if it is close or reveals a promising scaling trend
- `discard` if it uses savings inefficiently
- `crash` if it failed
- `invalid` if it changed too many things at once or broke the rules

## What Good Reinvestment Research Looks Like

Good reinvestment work is:
- based on one fixed quantization scheme
- budget-aware
- explicit about byte, time, and memory headroom
- one change at a time
- broad before deep
- willing to test bold ideas only after the simple scaling surface is understood

## Bottom Line

First freeze a winning CUDA quantization scheme.
Then treat its savings like a budget.
Spend that budget systematically:
- depth
- width
- feedforward capacity
- training compute
- then more adventurous ideas such as attention residuals and gated delta-style hybrids

The best reinvestment result is not the fanciest idea.
It is the one that uses the saved bytes and compute most efficiently under the real challenge constraints.
