# 2026-04-02 FP4 INT4 fp16-head

- Hypothesis: exact FP4 may trade quality and bytes differently from NF4 on the same INT4 grouped codec.
- Exact change made: set `SCALAR_QUANTIZER=fp4` on top of the INT4 group-64 fp16-head lzma setup.
- Why this is in scope: this is still pure post-training quantization and serialization.
- Expected effect on val_bpb: unknown versus NF4; this is a breadth comparison run.
- Expected effect on artifact bytes: broadly similar to NF4.
- Final result: pending.
- Short conclusion: pending.
