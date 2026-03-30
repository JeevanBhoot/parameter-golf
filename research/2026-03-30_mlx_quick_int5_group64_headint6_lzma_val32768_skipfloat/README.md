# 2026-03-30_mlx_quick_int5_group64_headint6_lzma_val32768_skipfloat

Head-bitwidth sweep run on the INT5 grouped-lzma frontier.

- Body quantization: INT5 grouped by 64 columns
- Tied embedding/output head: INT6 (`QMAX_HEAD=31`)
- AWQ: disabled
- Quick-track protocol: `VAL_SUBSET_SEQS=32768`, `SKIP_FINAL_FLOAT_VAL=1`

This run is scheduled through `research/run_mlx_experiment_queue.py`; the exact launch env lives in `research/queue_specs/2026-03-30_headbitsweep_queue.json`.
