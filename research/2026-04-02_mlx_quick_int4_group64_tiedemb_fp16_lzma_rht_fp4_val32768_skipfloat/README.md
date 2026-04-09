# 2026-04-02 RHT FP4 INT4 fp16-head

- Hypothesis: FP4 may pair differently with input rotation than NF4 did.
- Exact change made: set `SCALAR_QUANTIZER=fp4`, `INPUT_ROTATION=hadamard`, and `ROTATION_SEED=1337` on top of the INT4 group-64 fp16-head lzma setup.
- Why this is in scope: still weight-only PTQ plus deterministic rotation and serialization.
- Expected effect on val_bpb: breadth comparison against plain FP4 and rotated NF4.
- Expected effect on artifact bytes: similar to plain FP4 plus a small rotation-sign overhead.
- Final result: pending.
- Short conclusion: pending.
