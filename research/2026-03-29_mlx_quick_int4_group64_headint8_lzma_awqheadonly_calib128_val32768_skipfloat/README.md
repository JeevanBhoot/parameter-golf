# 2026-03-29_mlx_quick_int4_group64_headint8_lzma_awqheadonly_calib128_val32768_skipfloat

Structural head-only AWQ follow-up to the INT4 + group64 + INT8 tied-head + lzma branch.

- Body quantization: INT4 grouped by 64 columns
- Tied embedding/output head: INT8
- AWQ: enabled only for `tok_emb.weight`
- AWQ head calibration: 128 deterministic train-prefix sequences
- Quick-track protocol: `VAL_SUBSET_SEQS=32768`, `SKIP_FINAL_FLOAT_VAL=1`

This run is scheduled through `research/run_mlx_experiment_queue.py`; the exact launch env lives in `research/queue_specs/2026-03-29_headonly_int4_queue.json`.
