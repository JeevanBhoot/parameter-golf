# 2026-04-02 NF4 Sparse Residual INT4 fp16-head

- Hypothesis: NF4 already reduces dense quantization error; a tiny sparse residual budget may buy back more quality efficiently than it did for uniform INT4.
- Exact change made: set `SPARSE_RESIDUAL_FRAC=0.001` on top of the INT4 group-64 NF4 fp16-head quick-track setup.
- Why this is in scope: this is still pure compression and post-training quantization metadata.
- Expected effect on val_bpb: improve versus plain INT4 NF4.
- Expected effect on artifact bytes: small increase from sparse indices and fp16 residual values.
- Final result: pending.
- Short conclusion: pending.
