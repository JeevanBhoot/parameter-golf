# 2026-04-02 AF4 INT4 fp16-head

- Hypothesis: abnormal-float AF4 may be a stronger 4-bit codebook than FP4 and a useful contrast to NF4.
- Exact change made: set `SCALAR_QUANTIZER=af4` on top of the INT4 group-64 fp16-head lzma setup.
- Why this is in scope: this is still pure post-training quantization and serialization.
- Expected effect on val_bpb: breadth comparison against NF4 and FP4.
- Expected effect on artifact bytes: broadly similar to other 4-bit codebooks.
- Final result: pending.
- Short conclusion: pending.
