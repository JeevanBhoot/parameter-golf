# 2026-04-02 RHT NF4 Sparse Residual INT4 fp16-head

- Hypothesis: the rotated NF4 baseline is strongest so far in this family; a tiny sparse residual may push it further with limited byte cost.
- Exact change made: set `INPUT_ROTATION=hadamard`, `ROTATION_SEED=1337`, and `SPARSE_RESIDUAL_FRAC=0.001` on top of INT4 NF4 fp16-head.
- Why this is in scope: still PTQ plus deterministic compression metadata, with no training/eval changes.
- Expected effect on val_bpb: improve versus rotated INT4 NF4.
- Expected effect on artifact bytes: small increase from sparse correction storage.
- Final result: pending.
- Short conclusion: pending.
