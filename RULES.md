# RULES

`README.md` is canonical. This file only adds rule clarifications that are easy to miss. If this file and `README.md` disagree, `README.md` wins.

## A valid `val_bpb`

Evaluation must be all of the following:

- Strictly causal. At token `t`, the distribution may depend only on the artifact and tokens `< t`. No `x_t`, no future tokens, no future-derived statistics, and no external/runtime side information.
- Fully normalized. Before scoring `x_t`, define one normalized distribution over the full official token vocabulary. No "score only the realized token", residual fill-in, or bucket/expert-only normalization.
- Score-before-update. Score token `t` first; only then may any cache, adapter, optimizer, n-gram table, LoRA, etc. update using `x_t`.
- Single-pass. Exactly one left-to-right pass. No rescoring, no second pass, no retrospective changes, and no oracle `min()`/best-of selection across eval runs.

## Metric correctness

- `val_bpb` is bits per byte, not bits per token.
- Byte counts must come from the actual tokenizer piece table.
- For SentencePiece, handle leading-space pieces (`▁`), byte tokens, and zero-byte boundary/control tokens correctly.
- Score the full `fineweb_val_*` split in original shard/token order.
- No single-shard eval, fixed-number-of-batches eval, reordering, filtering, or cherry-picking.

## Training vs eval compute

- No compute transfer between phases.
- Any data-consuming step that changes model state before validation scoring counts as training compute, even if run after "training" ends.
- Examples: GPTQ/Hessian calibration, weight updates on unscored validation data, cache prefill from future tokens.
- Inference-only tricks are fine if they obey the rules above: sliding windows, KV caches, etc.
- Eval-time adaptation is only valid if it is causal and score-first: e.g. score-first TTT or causal n-gram caches built only from already-scored tokens.
