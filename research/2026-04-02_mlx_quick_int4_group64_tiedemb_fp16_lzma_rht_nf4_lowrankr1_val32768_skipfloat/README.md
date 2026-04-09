# 2026-04-02 RHT NF4 Low-Rank Residual INT4 fp16-head

- Hypothesis: the strongest NF4 baseline may benefit from a tiny rank-1 residual factorization even if the uniform-INT4 low-rank runs were weak.
- Exact change made: set `INPUT_ROTATION=hadamard`, `ROTATION_SEED=1337`, `LOWRANK_RESIDUAL_RANK=1`, and `LOWRANK_RESIDUAL_MODE=plain` on top of INT4 NF4 fp16-head.
- Why this is in scope: this is still pure PTQ plus compressed residual storage.
- Expected effect on val_bpb: improve versus rotated INT4 NF4.
- Expected effect on artifact bytes: moderate increase from low-rank factors.
- Final result: pending.
- Short conclusion: pending.
