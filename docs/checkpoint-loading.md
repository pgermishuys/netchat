# Checkpoint Loading

This directory contains utilities for loading pretrained weights into the nanochat-dotnet model.

## Overview

The `CheckpointLoader` class provides functionality to load PyTorch checkpoint files (.pt, .pth, .dat) into the C# GPT model.

## Usage

### Loading a TorchSharp Checkpoint

If you have a checkpoint that was saved from a C# TorchSharp model (matching naming conventions):

```csharp
using NanoChat.Core.Model;

var config = new GPTConfig { /* your config */ };
var model = new GPT(config);

// Load checkpoint directly
CheckpointLoader.LoadIntoModel(model, "model.dat");
```

### Loading a PyTorch (Python) Checkpoint

PyTorch checkpoints from Python use different naming conventions. You need to convert them first:

1. **Convert the checkpoint** using the provided Python script:
   ```bash
   python convert_checkpoint.py input.pt output.dat
   ```

2. **Load the converted checkpoint**:
   ```csharp
   CheckpointLoader.LoadIntoModel(model, "output.dat");
   ```

## Parameter Name Mapping

The `CheckpointLoader` handles mapping between PyTorch and TorchSharp naming conventions:

| PyTorch (Python)              | TorchSharp (C#)                |
|------------------------------|--------------------------------|
| `token_embedding.weight`      | `_tokenEmbedding.weight`       |
| `blocks.0.attn.q_proj.weight` | `block_0._attn._qProj.weight`  |
| `blocks.0.mlp.fc1.weight`     | `block_0._mlp._fc1.weight`     |
| `lm_head.weight`              | `_lmHead.weight`               |

## Conversion Script

The `convert_checkpoint.py` script converts PyTorch checkpoints to TorchSharp-compatible format:

```bash
# Basic usage
python convert_checkpoint.py input.pt output.dat

# Verbose output (shows all conversions)
python convert_checkpoint.py input.pt output.dat --verbose
```

The script:
- Loads PyTorch checkpoint files (.pt, .pth)
- Converts parameter names to TorchSharp conventions
- Saves as a TorchSharp-compatible checkpoint (.dat)
- Preserves all tensor data and model weights

## File Format

Both PyTorch and TorchSharp use the PyTorch serialization format (pickle + torch tensors). The main difference is in parameter naming conventions within the serialized state dictionary.

## Notes

- RMSNorm in this implementation has no learnable parameters, so norm weights are not loaded
- The model structure (number of layers, heads, embedding size) must match the checkpoint
- Use `strict: false` in `LoadIntoModel()` to allow missing/unexpected parameters (useful for debugging)
