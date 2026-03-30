# 2026-03-29_mlx_quick_int4_group64_tiedemb_fp16_lzma_awq_calib128_val32768_skipfloat

AWQ-style activation-aware column scaling on top of the INT4 group-of-64 + fp16 tied-head + lzma branch.

- Body quantization: INT4 grouped by 64 columns
- Tied embedding/output head: fp16 passthrough
- AWQ: full large-matrix body AWQ
- Calibration: 128 deterministic train-prefix sequences
- Quick-track protocol: `VAL_SUBSET_SEQS=32768`, `SKIP_FINAL_FLOAT_VAL=1`

This run is scheduled through `research/run_mlx_experiment_queue.py`; the exact launch env lives in `research/queue_specs/2026-03-29_headonly_int4_queue.json`.
