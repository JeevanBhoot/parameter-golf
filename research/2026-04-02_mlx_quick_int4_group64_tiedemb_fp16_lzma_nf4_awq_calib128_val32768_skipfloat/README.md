# 2026-04-02 NF4 AWQ INT4 fp16-head

- Hypothesis: activation-aware scaling may recover some of the remaining INT4 NF4 loss without giving back too many bytes.
- Exact change made: enable `AWQ_ENABLED=1` with `AWQ_CALIBRATION_SEQS=128` on top of the INT4 group-64 NF4 fp16-head quick-track setup.
- Why this is in scope: this is still pure post-training quantization plus deterministic calibration.
- Expected effect on val_bpb: improve versus plain INT4 NF4.
- Expected effect on artifact bytes: modest increase from stored AWQ scales.
- Final result: pending.
- Short conclusion: pending.
