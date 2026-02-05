# NanoChat .NET

A C#/.NET implementation of nanochat inference to deeply understand transformer architecture through building.

## Status

✅ **Phase 1: Project Foundation** - Complete

## Prerequisites

- .NET 9.0 SDK or later
- macOS users: `brew install libomp`

## Building

```bash
# Restore and build
dotnet build NanoChat.sln

# Build completes with 0 warnings and 0 errors
```

## Running

```bash
# Run the CLI application
dotnet run --project src/NanoChat.CLI/NanoChat.CLI.csproj

# Expected output:
# NanoChat - Testing TorchSharp Integration
# ==========================================
# Creating tensor... ✓ Success
# Running matmul... ✓ Success
#
# All TorchSharp tests passed!
```

## Testing

```bash
# Run all tests
dotnet test NanoChat.sln

# All tests pass (2/2 passing)
```

## Project Structure

```
├── src/
│   ├── NanoChat.Core/       # Core library with tensor operations
│   └── NanoChat.CLI/        # CLI application
├── tests/
│   └── NanoChat.Core.Tests/ # xUnit tests
├── docs/
│   └── PLAN.md             # Full project plan
└── progress.txt            # Progress log
```

## Dependencies

- **TorchSharp-cpu** v0.105.2 - PyTorch bindings for .NET
  - Includes native libtorch 2.7.1 CPU libraries
  - Cross-platform support (Windows, Linux, macOS)

## Next Phase

Feature 2: Tokenizer - Encode/decode text compatible with nanochat

See [docs/PLAN.md](docs/PLAN.md) for the full roadmap.
