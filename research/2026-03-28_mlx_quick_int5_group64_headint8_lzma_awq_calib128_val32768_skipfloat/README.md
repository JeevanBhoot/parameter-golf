# 2026-03-28_mlx_quick_int5_group64_headint8_lzma_awq_calib128_val32768_skipfloat

AWQ-style activation-aware column scaling on top of the current INT5 group-of-64 + INT8 tied-head + lzma frontier.

- Body quantization: INT5 grouped by 64 columns
- Tied embedding/output head: INT8
- AWQ calibration: 128 deterministic train-prefix sequences
- AWQ tied-head scaling: disabled
- Alpha search: `0.0,0.25,0.5,0.75,1.0`
- Quick-track protocol: `VAL_SUBSET_SEQS=32768`, `SKIP_FINAL_FLOAT_VAL=1`

This run is scheduled through `research/run_mlx_experiment_queue.py`; the exact launch env lives in `research/queue_specs/2026-03-28_awq_queue.json`.
