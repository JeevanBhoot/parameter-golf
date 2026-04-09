# 2026-04-02 RHT NF4 AWQ INT4 fp16-head

- Hypothesis: the best NF4 variant so far is rotated; adding AWQ on top may recover further quality.
- Exact change made: enable `INPUT_ROTATION=hadamard`, `ROTATION_SEED=1337`, and `AWQ_ENABLED=1` with `AWQ_CALIBRATION_SEQS=128` on top of INT4 NF4 fp16-head.
- Why this is in scope: this remains weight-only PTQ with deterministic calibration and serialization changes only.
- Expected effect on val_bpb: improve versus rotated INT4 NF4.
- Expected effect on artifact bytes: modest increase from AWQ scales.
- Final result: pending.
- Short conclusion: pending.
