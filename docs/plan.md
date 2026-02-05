# nanochat-dotnet Project Plan

## Goal

Build a C#/.NET implementation of nanochat inference to deeply understand transformer architecture through building.

## Target

- **.NET Version:** .NET 10 (Latest LTS)
- **Approach:** Libraries first → Replace with own implementations

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Tokenizer  │───▶│   GPT Model  │───▶│    Engine    │
│              │    │              │    │  (Inference) │
│ • Encode     │    │ • Embedding  │    │ • KV Cache   │
│ • Decode     │    │ • Attention  │    │ • Generate   │
│ • Special    │    │ • MLP        │    │ • Sample     │
│   tokens     │    │ • Blocks     │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────┐
│                    Chat CLI                          │
└──────────────────────────────────────────────────────┘
```

## Phases

### Phase 1: Libraries (Get it Working)

Use existing libraries to get a working implementation quickly.

| Component | Library |
|-----------|---------|
| Tensors & Ops | TorchSharp-cpu |
| Tokenizer | Tiktoken (custom loader) |
| Weight Loading | TorchSharp (PyTorch .pt files) |

### Phase 2: Replace (Understand It)

Replace dependencies with own implementations in order of complexity:

1. Tokenizer → Own BPE implementation
2. RMSNorm → Simple, no learnable params
3. RoPE → Rotary embeddings from scratch
4. Attention → Scaled dot-product, causal mask
5. MLP → Linear layers, ReLU²
6. Tensor Ops → Own matmul, softmax, etc.

---

## Features & User Stories

### Feature 1: Project Foundation

**Goal:** Solution structure, dependencies, builds successfully

| Story | Description | Acceptance Criteria | Status |
|-------|-------------|---------------------|--------|
| 1.1 | Create solution and project structure | `dotnet build` succeeds | ✅ |
| 1.2 | Add TorchSharp-cpu dependency | Can create tensor, run matmul | ✅ |
| 1.3 | Add test project | `dotnet test` runs | ✅ |

---

### Feature 2: Tokenizer

**Goal:** Encode/decode text compatible with nanochat

| Story | Description | Acceptance Criteria | Status |
|-------|-------------|---------------------|--------|
| 2.1 | Create tokenizer interface | `ITokenizer` with Encode/Decode | ✅ |
| 2.2 | Implement tiktoken-based tokenizer | Load mergeable_ranks, encode text | ✅ |
| 2.3 | Add special token support | Encode `<\|bos\|>`, `<\|user_start\|>`, etc. | ✅ |
| 2.4 | Load nanochat tokenizer from disk | Read `tokenizer.pkl` format | ✅ |
| 2.5 | Test against Python implementation | Same input → same token IDs | ✅ |

---

### Feature 3: Model Components

**Goal:** Implement all GPT building blocks

| Story | Description | Acceptance Criteria | Status |
|-------|-------------|---------------------|--------|
| 3.1 | Implement RMSNorm | `norm(x)` matches Python output | ✅ |
| 3.2 | Implement Rotary Embeddings | Precompute cos/sin, apply to Q/K | ✅ |
| 3.3 | Implement Multi-Head Attention | Causal self-attention works | ✅ |
| 3.4 | Implement GQA (Group-Query Attention) | n_kv_head < n_head works | ✅ |
| 3.5 | Implement MLP block | ReLU² activation | ✅ |
| 3.6 | Implement Transformer Block | Attention + MLP + residuals | ✅ |
| 3.7 | Implement Value Embeddings | ResFormer-style alternating VE | ✅ |

---

### Feature 4: GPT Model

**Goal:** Full model that can compute logits

| Story | Description | Acceptance Criteria | Status |
|-------|-------------|---------------------|--------|
| 4.1 | Implement GPTConfig | Dataclass with all hyperparams | ✅ |
| 4.2 | Implement GPT model shell | Embeddings, blocks, lm_head | ✅ |
| 4.3 | Implement forward pass | Input tokens → logits | ✅ |
| 4.4 | Add softcap to logits | `15 * tanh(logits / 15)` | ✅ |
| 4.5 | Add sliding window support | Per-layer window sizes | ✅ |

---

### Feature 5: Weight Loading

**Goal:** Load pretrained nanochat checkpoint

| Story | Description | Acceptance Criteria | Status |
|-------|-------------|---------------------|--------|
| 5.1 | Parse PyTorch checkpoint format | Read `.pt` file structure | ✅ |
| 5.2 | Map weight names to model | Handle naming differences | ✅ |
| 5.3 | Load weights into model | All parameters populated | ✅ |
| 5.4 | Verify loaded weights | Forward pass matches Python | ✅ |

---

### Feature 6: Inference Engine

**Goal:** Generate text autoregressively

| Story | Description | Acceptance Criteria | Status |
|-------|-------------|---------------------|--------|
| 6.1 | Implement naive generation | No cache, works correctly | ✅ |
| 6.2 | Implement temperature sampling | Adjustable randomness | ✅ |
| 6.3 | Implement top-k sampling | Filter low-probability tokens | ✅ |
| 6.4 | Implement KV-Cache | KVCache class with tests | ✅ |
| 6.5 | Integrate KV-Cache into generation | Cached inference works | ✅ |
| 6.6 | Optimize generation loop | Streaming token output | ⬜ |

---

### Feature 7: Chat Interface

**Goal:** Interactive CLI chat

| Story | Description | Acceptance Criteria | Status |
|-------|-------------|---------------------|--------|
| 7.1 | Implement conversation rendering | Format messages with special tokens | ⬜ |
| 7.2 | Implement CLI input loop | Read user input | ⬜ |
| 7.3 | Implement streaming output | Tokens appear as generated | ⬜ |
| 7.4 | Handle conversation history | Multi-turn works | ⬜ |

---

## nanochat-Specific Implementation Details

Key details from the Python implementation that must be matched:

| Aspect | Implementation |
|--------|----------------|
| Norm | `F.rms_norm()` - no learnable parameters |
| Activation | `ReLU²` (relu then square) |
| Position Encoding | Rotary (RoPE), base=10000 |
| QK Norm | Applied after RoPE |
| Attention | GQA support, sliding windows |
| Logit Softcap | `15 * tanh(x/15)` |
| Residual | Per-layer `resid_lambdas` and `x0_lambdas` |
| Value Embeddings | Alternating layers, gated |
| Vocab Size | 32768 (padded to multiple of 64) |
| Sequence Length | 2048 |

## GPTConfig Defaults

```
sequence_len: 2048
vocab_size: 32768
n_layer: 12
n_head: 6
n_kv_head: 6
n_embd: 768
window_pattern: "SSSL"
```

---

## References

- [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [RMSNorm Paper](https://arxiv.org/abs/1910.07467)
- [TorchSharp NuGet](https://www.nuget.org/packages/TorchSharp-cpu)
- [Tiktoken for .NET](https://github.com/microsoft/Tokenizer)

---

## Status Legend

- ⬜ Not started
- 🟡 In progress
- ✅ Complete
- ❌ Blocked
