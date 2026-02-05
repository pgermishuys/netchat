using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NanoChat.Core.Model;

/// <summary>
/// Rotary Position Embedding (RoPE)
/// 
/// Applies rotary position embeddings to queries and keys in attention.
/// RoPE encodes position information by rotating Q and K vectors in a way
/// that preserves relative position information.
/// 
/// Reference: https://arxiv.org/abs/2104.09864
/// </summary>
public class RotaryEmbedding : Module<Tensor, long, Tensor>
{
    private readonly int _dim;
    private readonly int _maxSeqLen;
    private readonly double _base;
    private Tensor? _cosCache;
    private Tensor? _sinCache;

    /// <summary>
    /// Initialize Rotary Position Embedding
    /// </summary>
    /// <param name="dim">Dimension per head (head_dim)</param>
    /// <param name="maxSeqLen">Maximum sequence length</param>
    /// <param name="base">Base for frequency computation (default: 10000.0)</param>
    /// <param name="name">Module name</param>
    public RotaryEmbedding(int dim, int maxSeqLen = 2048, double @base = 10000.0, string? name = null) 
        : base(name ?? "RotaryEmbedding")
    {
        _dim = dim;
        _maxSeqLen = maxSeqLen;
        _base = @base;
        
        // Precompute cos and sin tables
        _buildCache();
    }

    /// <summary>
    /// Precompute cos and sin rotation matrices for all positions
    /// </summary>
    private void _buildCache()
    {
        // Compute inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
        // Shape: (dim/2,)
        var invFreq = tensor(
            Enumerable.Range(0, _dim / 2)
                .Select(i => (float)(1.0 / Math.Pow(_base, (2.0 * i) / _dim)))
                .ToArray()
        );

        // Compute position indices: [0, 1, 2, ..., maxSeqLen-1]
        // Shape: (maxSeqLen,)
        var t = arange(_maxSeqLen, dtype: ScalarType.Float32);

        // Compute angles: outer product of positions and frequencies
        // Shape: (maxSeqLen, dim/2)
        var freqs = outer(t, invFreq);

        // Concatenate freqs with itself to get full dimension
        // This allows us to apply rotation to pairs of dimensions
        // Shape: (maxSeqLen, dim)
        var emb = cat(new[] { freqs, freqs }, dim: -1);

        // Precompute cos and sin
        // Shape: (maxSeqLen, dim)
        _cosCache = emb.cos();
        _sinCache = emb.sin();

        // Register as buffers (not trainable parameters)
        register_buffer("cos_cache", _cosCache);
        register_buffer("sin_cache", _sinCache);
    }

    /// <summary>
    /// Apply rotary embeddings to input tensor
    /// </summary>
    /// <param name="x">Input tensor of shape (batch, seqLen, nHeads, headDim)</param>
    /// <param name="seqLen">Sequence length to use from cache</param>
    /// <returns>Rotated tensor of same shape as input</returns>
    public override Tensor forward(Tensor x, long seqLen)
    {
        if (_cosCache is null)
        {
            throw new InvalidOperationException("Cosine cache not initialized");
        }
        if (_sinCache is null)
        {
            throw new InvalidOperationException("Sine cache not initialized");
        }

        // Get cos and sin for the current sequence length
        // Shape: (seqLen, headDim)
        var cos = _cosCache[TensorIndex.Slice(0, seqLen)];
        var sin = _sinCache[TensorIndex.Slice(0, seqLen)];

        // Reshape for broadcasting: (1, seqLen, 1, headDim)
        cos = cos.unsqueeze(0).unsqueeze(2);
        sin = sin.unsqueeze(0).unsqueeze(2);

        // Apply rotation
        // Split input into two halves and apply rotary transformation
        // The RoPE formula: rotate_half where we negate and swap the second half
        var x1 = x[TensorIndex.Ellipsis, TensorIndex.Slice(null, _dim / 2)];
        var x2 = x[TensorIndex.Ellipsis, TensorIndex.Slice(_dim / 2, null)];

        // Get corresponding cos/sin for each half
        var cos1 = cos[TensorIndex.Ellipsis, TensorIndex.Slice(null, _dim / 2)];
        var cos2 = cos[TensorIndex.Ellipsis, TensorIndex.Slice(_dim / 2, null)];
        var sin1 = sin[TensorIndex.Ellipsis, TensorIndex.Slice(null, _dim / 2)];
        var sin2 = sin[TensorIndex.Ellipsis, TensorIndex.Slice(_dim / 2, null)];

        // Rotary transformation:
        // The standard RoPE applies: [x1*cos1 - x2*sin1, x2*cos2 + x1*sin2]
        // But since cos1==cos2 and sin1==sin2 (from how we built the cache), it simplifies to:
        // [x1*cos - x2*sin, x2*cos + x1*sin]
        var rotated1 = x1 * cos1 - x2 * sin1;
        var rotated2 = x2 * cos2 + x1 * sin2;

        // Concatenate back
        return cat(new[] { rotated1, rotated2 }, dim: -1);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _cosCache?.Dispose();
            _sinCache?.Dispose();
        }
        base.Dispose(disposing);
    }
}
