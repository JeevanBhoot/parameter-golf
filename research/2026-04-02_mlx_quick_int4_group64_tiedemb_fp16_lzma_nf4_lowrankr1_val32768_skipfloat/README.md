# 2026-04-02 NF4 Low-Rank Residual INT4 fp16-head

- Hypothesis: rank-1 low-rank reconstruction may be more useful once the base INT4 quantizer is NF4 rather than uniform.
- Exact change made: set `LOWRANK_RESIDUAL_RANK=1` and `LOWRANK_RESIDUAL_MODE=plain` on top of the INT4 group-64 NF4 fp16-head quick-track setup.
- Why this is in scope: this is still post-training quantization plus compressed residual metadata.
- Expected effect on val_bpb: improve versus plain INT4 NF4.
- Expected effect on artifact bytes: moderate increase from the stored low-rank factors.
- Final result: pending.
- Short conclusion: pending.
