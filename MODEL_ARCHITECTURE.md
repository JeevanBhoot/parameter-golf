# Model Architecture

This document describes the shared model architecture used by [train_gpt_mlx.py](/Users/jeebho01/parameter-golf/train_gpt_mlx.py) and [train_gpt.py](/Users/jeebho01/parameter-golf/train_gpt.py).

## Architecture Summary

- Model family: autoregressive causal LM with a split encoder/decoder-style transformer trunk rather than a plain linear decoder stack.
- Default shape: `vocab_size=1024`, `d_model=512`, `num_layers=9`, `num_heads=8`, `num_kv_heads=4`, `head_dim=64`, `mlp_hidden=1024`, `rope_base=10000`.
- Stem: token embedding only, then RMSNorm; there is no learned positional embedding.
- Core trunk: 4 encoder blocks store skips, then 5 decoder blocks consume those skips in reverse order with learned per-channel `skip_weights`.
- Each block first mixes the current state with the original stem state `x0` through a learned 2-way per-channel `resid_mix`, then applies pre-norm attention and pre-norm MLP residual updates.
- Attention is GQA: `Q` has 8 heads, `K/V` have 4 heads, with RMSNorm on `Q/K`, RoPE on `Q/K`, and a learned per-head `q_gain`.
- Output head: final RMSNorm, then tied embedding projection to logits plus tanh softcap (`c=30`) before cross-entropy in both implementations.

## Assumptions

- This document treats the tied LM head as part of the full architecture even though [train_gpt_mlx.py](/Users/jeebho01/parameter-golf/train_gpt_mlx.py) returns hidden states from `__call__` and applies the logits plus loss path separately in `loss()`.
- The MLX and PyTorch implementations match architecturally; the PyTorch file is used here only to confirm the same output-head structure.

## Whole Network

```mermaid
flowchart TD
    IN["Input token ids<br/>[B, T]"] --> EMB["Token Embedding<br/>1024 -> 512"]
    EMB --> STEM["RMSNorm stem<br/>x0 = x<br/>[B, T, 512]"]

    subgraph ENC[Encoder half: 4 blocks]
        E1["Block 1"]
        E2["Block 2"]
        E3["Block 3"]
        E4["Block 4"]
        E1 --> E2 --> E3 --> E4
    end

    subgraph DEC[Decoder half: 5 blocks]
        A1["Add skip_weight[0] * skip(E4)"]
        D1["Block 5"]
        A2["Add skip_weight[1] * skip(E3)"]
        D2["Block 6"]
        A3["Add skip_weight[2] * skip(E2)"]
        D3["Block 7"]
        A4["Add skip_weight[3] * skip(E1)"]
        D4["Block 8"]
        D5["Block 9<br/>no encoder skip left"]
        A1 --> D1 --> A2 --> D2 --> A3 --> D3 --> A4 --> D4 --> D5
    end

    STEM --> E1
    E4 --> A1
    E3 -. reversed skip .-> A2
    E2 -. reversed skip .-> A3
    E1 -. reversed skip .-> A4

    D5 --> FN["Final RMSNorm<br/>[B, T, 512]"]
    FN --> H["Hidden states<br/>[B, T, 512]"]
    H --> LM["Tied LM head<br/>W_E^T : 512 -> 1024"]
    LM --> SC["Softcap<br/>30 * tanh(logits / 30)"]
    SC --> OUT["Token logits<br/>[B, T, 1024]"]
    OUT --> CE["Cross-entropy loss"]
```

## Block Diagrams

### Transformer Block

```mermaid
flowchart TD
    X["Input x<br/>[B, T, 512]"] --> MIX["resid_mix<br/>x_mix = a * x + b * x0"]
    X0["Stem state x0<br/>[B, T, 512]"] --> MIX

    MIX --> AN["RMSNorm"]
    AN --> ATTN["CausalSelfAttention"]
    ATTN --> ADD1["Residual add<br/>x_mix + attn_scale * attn_out"]

    ADD1 --> MN["RMSNorm"]
    MN --> MLP["MLP<br/>512 -> 1024 -> 512<br/>ReLU^2"]
    MLP --> ADD2["Residual add<br/>prev + mlp_scale * mlp_out"]

    ADD2 --> Y["Output<br/>[B, T, 512]"]
```

### Causal Self-Attention

```mermaid
flowchart TD
    X["Input<br/>[B, T, 512]"] --> QP["Q projection<br/>512 -> 8 x 64"]
    X --> KP["K projection<br/>512 -> 4 x 64"]
    X --> VP["V projection<br/>512 -> 4 x 64"]

    QP --> QN["RMSNorm on Q"]
    KP --> KN["RMSNorm on K"]

    QN --> QR["RoPE on Q"]
    KN --> KR["RoPE on K"]

    QR --> QG["q_gain<br/>per-head scale"]
    KR --> SDPA
    VP --> SDPA["Scaled dot-product attention<br/>causal, GQA"]

    QG --> SDPA
    SDPA --> MERGE["Merge heads<br/>[B, T, 512]"]
    MERGE --> OP["Output projection<br/>512 -> 512"]
    OP --> Y["attn_out<br/>[B, T, 512]"]
```

## Implementation Spec

| Block | Order | Key parameters | Input shape | Output shape | Notes |
| --- | --- | --- | --- | --- | --- |
| Token embedding + stem norm | 1 | `vocab=1024`, `d_model=512` | `[B,T]` | `[B,T,512]` | Stem output is also saved as `x0` and reused by every block. |
| Encoder stack | 2 | `4 x Block` | `[B,T,512]` | `[B,T,512]` | Outputs of all 4 encoder blocks are pushed onto a skip stack. |
| Transformer block | repeated | `resid_mix`, `attn_scale`, `mlp_scale` are per-channel vectors of size `512` | `[B,T,512]`, `x0` | `[B,T,512]` | Pre-norm attention and pre-norm MLP, both residual. |
| Attention sublayer | inside block | `8` query heads, `4` KV heads, `head_dim=64`, RoPE, learned `q_gain` | `[B,T,512]` | `[B,T,512]` | GQA causal attention with RMSNorm on `Q` and `K` before RoPE. |
| MLP sublayer | inside block | `512 -> 1024 -> 512`, `mlp_mult=2` | `[B,T,512]` | `[B,T,512]` | Activation is `ReLU^2`, not GELU/SiLU. |
| Decoder stack | 3 | `5 x Block` | `[B,T,512]` | `[B,T,512]` | First 4 decoder blocks each add one reversed encoder skip via learned `skip_weights`; last decoder block has no skip input. |
| Final norm | 4 | RMSNorm | `[B,T,512]` | `[B,T,512]` | Returned directly by MLX `__call__`. |
| LM head | 5 | tied embedding projection `W_E^T`, softcap `c=30` | `[B,T,512]` | `[B,T,1024]` | In MLX this lives in `loss()`; in PyTorch it is part of `forward()`. |

## Code Anchors

- Hyperparameters and default model shape: [train_gpt_mlx.py:39](/Users/jeebho01/parameter-golf/train_gpt_mlx.py#L39)
- MLX attention block: [train_gpt_mlx.py:291](/Users/jeebho01/parameter-golf/train_gpt_mlx.py#L291)
- MLX transformer block: [train_gpt_mlx.py:350](/Users/jeebho01/parameter-golf/train_gpt_mlx.py#L350)
- MLX GPT trunk and skip flow: [train_gpt_mlx.py:378](/Users/jeebho01/parameter-golf/train_gpt_mlx.py#L378)
- MLX logits and loss path: [train_gpt_mlx.py:431](/Users/jeebho01/parameter-golf/train_gpt_mlx.py#L431)
- PyTorch attention block: [train_gpt.py:555](/Users/jeebho01/parameter-golf/train_gpt.py#L555)
- PyTorch GPT forward and tied head: [train_gpt.py:648](/Users/jeebho01/parameter-golf/train_gpt.py#L648)
