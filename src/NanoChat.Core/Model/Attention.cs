using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NanoChat.Core.Model;

/// <summary>
/// Multi-Head Causal Self-Attention with Group-Query Attention (GQA) support
/// 
/// Implements scaled dot-product attention with the following features:
/// - Group-Query Attention (GQA): n_kv_head can be less than n_head
/// - Causal masking to prevent attending to future tokens
/// - QK normalization applied after RoPE
/// - Rotary position embeddings
/// 
/// Reference: https://arxiv.org/abs/1706.03762 (Attention Is All You Need)
/// </summary>
public class Attention : Module<Tensor, long, Tensor>
{
    private readonly int _nEmbd;
    private readonly int _nHead;
    private readonly int _nKvHead;
    private readonly int _headDim;
    private readonly double _scale;
    private readonly int? _windowSize;
    
    private readonly Module<Tensor, Tensor> _qProj;
    private readonly Module<Tensor, Tensor> _kProj;
    private readonly Module<Tensor, Tensor> _vProj;
    private readonly Module<Tensor, Tensor> _outProj;
    private readonly RMSNorm _qNorm;
    private readonly RMSNorm _kNorm;
    private readonly RotaryEmbedding _rope;
    private readonly ValueEmbedding? _valueEmbedding;

    /// <summary>
    /// Initialize Multi-Head Attention layer
    /// </summary>
    /// <param name="nEmbd">Embedding dimension</param>
    /// <param name="nHead">Number of attention heads</param>
    /// <param name="nKvHead">Number of key-value heads (for GQA). If null, defaults to nHead</param>
    /// <param name="windowSize">Sliding window size for attention. If null, use full attention</param>
    /// <param name="maxSeqLen">Maximum sequence length for RoPE</param>
    /// <param name="ropeBase">Base for RoPE frequency computation</param>
    /// <param name="useValueEmbedding">Whether to use ResFormer-style value embeddings</param>
    /// <param name="name">Module name</param>
    public Attention(
        int nEmbd, 
        int nHead, 
        int? nKvHead = null, 
        int? windowSize = null,
        int maxSeqLen = 2048,
        double ropeBase = 10000.0,
        bool useValueEmbedding = false,
        string? name = null) 
        : base(name ?? "Attention")
    {
        _nEmbd = nEmbd;
        _nHead = nHead;
        _nKvHead = nKvHead ?? nHead;
        _windowSize = windowSize;
        
        if (_nEmbd % _nHead != 0)
        {
            throw new ArgumentException($"nEmbd ({_nEmbd}) must be divisible by nHead ({_nHead})");
        }
        
        if (_nHead % _nKvHead != 0)
        {
            throw new ArgumentException($"nHead ({_nHead}) must be divisible by nKvHead ({_nKvHead})");
        }
        
        _headDim = _nEmbd / _nHead;
        _scale = 1.0 / Math.Sqrt(_headDim);
        
        // Q projection has nHead heads
        _qProj = Linear(_nEmbd, _nHead * _headDim, hasBias: false);
        
        // K and V projections have nKvHead heads (for GQA)
        _kProj = Linear(_nEmbd, _nKvHead * _headDim, hasBias: false);
        _vProj = Linear(_nEmbd, _nKvHead * _headDim, hasBias: false);
        
        // Output projection
        _outProj = Linear(_nHead * _headDim, _nEmbd, hasBias: false);
        
        // QK normalization (applied after RoPE)
        _qNorm = new RMSNorm();
        _kNorm = new RMSNorm();
        
        // Rotary position embeddings
        _rope = new RotaryEmbedding(_headDim, maxSeqLen, ropeBase);
        
        // Value embeddings (optional, for ResFormer-style alternating layers)
        if (useValueEmbedding)
        {
            _valueEmbedding = new ValueEmbedding(_headDim, maxSeqLen);
        }
        
        // Register modules
        RegisterComponents();
    }

    /// <summary>
    /// Apply attention to input tensor
    /// </summary>
    /// <param name="x">Input tensor of shape (batch, seqLen, nEmbd)</param>
    /// <param name="seqLen">Sequence length (used for RoPE)</param>
    /// <returns>Output tensor of shape (batch, seqLen, nEmbd)</returns>
    public override Tensor forward(Tensor x, long seqLen)
    {
        return ForwardWithCache(x, seqLen, null, -1).output;
    }
    
    /// <summary>
    /// Apply attention to input tensor with optional KV caching for inference optimization.
    /// </summary>
    /// <param name="x">Input tensor of shape (batch, seqLen, nEmbd)</param>
    /// <param name="seqLen">Total sequence length including cached tokens (used for RoPE position encoding)</param>
    /// <param name="cache">Optional KV cache for efficient autoregressive generation</param>
    /// <param name="layerIdx">Layer index for cache lookup (required if cache is provided)</param>
    /// <returns>Tuple of (output, keys, values) where keys and values are before GQA expansion for caching</returns>
    public (Tensor output, Tensor keys, Tensor values) ForwardWithCache(
        Tensor x, 
        long seqLen, 
        KVCache? cache = null, 
        int layerIdx = -1)
    {
        if (cache is not null && layerIdx < 0)
            throw new ArgumentException("layerIdx must be provided when cache is not null");
        
        var batchSize = x.shape[0];
        var seqLength = x.shape[1];
        
        // Project to Q, K, V
        // Q: (batch, seqLen, nHead * headDim)
        // K, V: (batch, seqLen, nKvHead * headDim)
        var q = _qProj.forward(x);
        var k = _kProj.forward(x);
        var v = _vProj.forward(x);
        
        // Reshape to separate heads
        // Q: (batch, seqLen, nHead, headDim)
        // K, V: (batch, seqLen, nKvHead, headDim)
        q = q.view(batchSize, seqLength, _nHead, _headDim);
        k = k.view(batchSize, seqLength, _nKvHead, _headDim);
        v = v.view(batchSize, seqLength, _nKvHead, _headDim);
        
        // Add value embeddings if enabled (ResFormer-style)
        // This adds positional information directly to the value vectors
        if (_valueEmbedding is not null)
        {
            var ve = _valueEmbedding.forward(v, seqLen);
            v = v + ve;
        }
        
        // Apply RoPE to Q and K
        // When using cache: seqLen is TOTAL length (cached + new), seqLength is NEW tokens only
        // We need to apply RoPE for positions [cacheLen, seqLen)
        long cacheLen = cache?.CacheLength ?? 0;
        if (cacheLen > 0)
        {
            // With cache: apply RoPE with position offset
            q = _rope.ForwardWithOffset(q, cacheLen, seqLen);
            k = _rope.ForwardWithOffset(k, cacheLen, seqLen);
        }
        else
        {
            // No cache: apply RoPE normally for positions [0:seqLen]
            q = _rope.forward(q, seqLen);
            k = _rope.forward(k, seqLen);
        }
        
        // Apply QK normalization (after RoPE, as per nanochat spec)
        // Normalize over the head dimension (last dim)
        q = _qNorm.forward(q);
        k = _kNorm.forward(k);
        
        // Transpose for attention computation
        // Q: (batch, nHead, seqLen, headDim)
        // K, V: (batch, nKvHead, seqLen, headDim)
        q = q.transpose(1, 2);
        var kTransposed = k.transpose(1, 2);
        var vTransposed = v.transpose(1, 2);
        
        // Store K and V before GQA expansion for caching
        // These are in (batch, nKvHead, newSeqLen, headDim) format
        var kForCache = kTransposed.clone();
        var vForCache = vTransposed.clone();
        
        // Update or retrieve from cache if provided
        if (cache is not null)
        {
            var (cachedK, cachedV) = cache.Update(layerIdx, kForCache, vForCache);
            kTransposed.Dispose();
            vTransposed.Dispose();
            kTransposed = cachedK;
            vTransposed = cachedV;
            
            // Update seqLength to reflect total cached length
            seqLength = kTransposed.shape[2];
        }
        
        // Handle GQA: expand K and V to match Q's number of heads
        var kExpanded = kTransposed;
        var vExpanded = vTransposed;
        
        if (_nKvHead < _nHead)
        {
            var nRep = _nHead / _nKvHead;
            // Repeat each KV head nRep times
            // (batch, nKvHead, seqLen, headDim) -> (batch, nKvHead, nRep, seqLen, headDim)
            kExpanded = kTransposed.unsqueeze(2).expand(new long[] { batchSize, _nKvHead, nRep, seqLength, _headDim });
            vExpanded = vTransposed.unsqueeze(2).expand(new long[] { batchSize, _nKvHead, nRep, seqLength, _headDim });
            
            // Reshape to (batch, nHead, seqLen, headDim)
            kExpanded = kExpanded.reshape(batchSize, _nHead, seqLength, _headDim);
            vExpanded = vExpanded.reshape(batchSize, _nHead, seqLength, _headDim);
        }
        
        // Compute attention scores
        // Q: (batch, nHead, queryLen, headDim) 
        // K: (batch, nHead, seqLen, headDim)
        // scores: (batch, nHead, queryLen, seqLen)
        var queryLen = q.shape[2];
        var scores = matmul(q, kExpanded.transpose(-2, -1)) * _scale;
        
        // Apply causal mask: prevent attending to future positions
        // Create a mask where mask[i, j] = -inf if i < j (position i cannot attend to position j > i)
        // When using cache, queryLen might be 1 (generating single token) but seqLen includes all cached tokens
        var causalMask = ones(queryLen, seqLength, dtype: x.dtype, device: x.device)
            .tril(diagonal: seqLength - queryLen)  // Allow attending to all past + current position
            .log();  // Convert 1 -> 0, 0 -> -inf
        
        // Apply sliding window mask if specified
        if (_windowSize.HasValue)
        {
            // Create a band matrix: only allow attention within window
            var windowMask = ones(queryLen, seqLength, dtype: x.dtype, device: x.device)
                .triu(-_windowSize.Value)  // Upper triangular with offset
                .tril(diagonal: seqLength - queryLen)  // Lower triangular
                .log();
            
            // Combine causal and window masks (take maximum to keep valid positions)
            causalMask = maximum(causalMask, windowMask);
        }
        
        // Add mask to scores (broadcasting over batch and head dimensions)
        scores = scores + causalMask;
        
        // Apply softmax to get attention weights
        // (batch, nHead, queryLen, seqLen)
        var attnWeights = softmax(scores, dim: -1);
        
        // Apply attention weights to values
        // (batch, nHead, queryLen, seqLen) @ (batch, nHead, seqLen, headDim)
        // -> (batch, nHead, queryLen, headDim)
        var output = matmul(attnWeights, vExpanded);
        
        // Transpose back and reshape
        // (batch, nHead, queryLen, headDim) -> (batch, queryLen, nHead, headDim)
        output = output.transpose(1, 2);
        
        // Concatenate heads
        // (batch, queryLen, nHead, headDim) -> (batch, queryLen, nHead * headDim)
        output = output.contiguous().view(batchSize, queryLen, _nHead * _headDim);
        
        // Final output projection
        output = _outProj.forward(output);
        
        return (output, kForCache, vForCache);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _qProj?.Dispose();
            _kProj?.Dispose();
            _vProj?.Dispose();
            _outProj?.Dispose();
            _qNorm?.Dispose();
            _kNorm?.Dispose();
            _rope?.Dispose();
            _valueEmbedding?.Dispose();
        }
        base.Dispose(disposing);
    }
}
