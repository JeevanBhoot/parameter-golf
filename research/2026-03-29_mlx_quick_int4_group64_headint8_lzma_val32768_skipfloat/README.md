# 2026-03-29_mlx_quick_int4_group64_headint8_lzma_val32768_skipfloat

Plain INT4 grouped-lzma baseline with an INT8 tied head for the new head-only AWQ branch.

- Body quantization: INT4 grouped by 64 columns
- Tied embedding/output head: INT8
- AWQ: disabled
- Quick-track protocol: `VAL_SUBSET_SEQS=32768`, `SKIP_FINAL_FLOAT_VAL=1`

This run is scheduled through `research/run_mlx_experiment_queue.py`; the exact launch env lives in `research/queue_specs/2026-03-29_headonly_int4_queue.json`.
