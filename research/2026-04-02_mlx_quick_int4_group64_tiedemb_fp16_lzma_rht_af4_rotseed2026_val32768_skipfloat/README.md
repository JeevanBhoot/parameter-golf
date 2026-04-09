# 2026-04-02 RHT AF4 INT4 fp16-head

- Hypothesis: if AF4 benefits from incoherence similarly to some other low-bit codecs, rotation may sharpen it further.
- Exact change made: set `SCALAR_QUANTIZER=af4`, `INPUT_ROTATION=hadamard`, and `ROTATION_SEED=1337` on top of the INT4 group-64 fp16-head lzma setup.
- Why this is in scope: still weight-only PTQ plus deterministic rotation and serialization.
- Expected effect on val_bpb: breadth comparison against plain AF4 and the NF4 family.
- Expected effect on artifact bytes: similar to plain AF4 plus a small rotation-sign overhead.
- Final result: pending.
- Short conclusion: pending.
