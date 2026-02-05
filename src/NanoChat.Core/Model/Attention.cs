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

    /// <summary>
    /// Initialize Multi-Head Attention layer
    /// </summary>
    /// <param name="nEmbd">Embedding dimension</param>
    /// <param name="nHead">Number of attention heads</param>
    /// <param name="nKvHead">Number of key-value heads (for GQA). If null, defaults to nHead</param>
    /// <param name="windowSize">Sliding window size for attention. If null, use full attention</param>
    /// <param name="maxSeqLen">Maximum sequence length for RoPE</param>
    /// <param name="ropeBase">Base for RoPE frequency computation</param>
    /// <param name="name">Module name</param>
    public Attention(
        int nEmbd, 
        int nHead, 
        int? nKvHead = null, 
        int? windowSize = null,
        int maxSeqLen = 2048,
        double ropeBase = 10000.0,
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
        
        // Apply RoPE to Q and K
        q = _rope.forward(q, seqLen);
        k = _rope.forward(k, seqLen);
        
        // Apply QK normalization (after RoPE, as per nanochat spec)
        // Normalize over the head dimension (last dim)
        q = _qNorm.forward(q);
        k = _kNorm.forward(k);
        
        // Transpose for attention computation
        // Q: (batch, nHead, seqLen, headDim)
        // K, V: (batch, nKvHead, seqLen, headDim)
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);
        
        // Handle GQA: expand K and V to match Q's number of heads
        if (_nKvHead < _nHead)
        {
            var nRep = _nHead / _nKvHead;
            // Repeat each KV head nRep times
            // (batch, nKvHead, seqLen, headDim) -> (batch, nKvHead, nRep, seqLen, headDim)
            k = k.unsqueeze(2).expand(new long[] { batchSize, _nKvHead, nRep, seqLength, _headDim });
            v = v.unsqueeze(2).expand(new long[] { batchSize, _nKvHead, nRep, seqLength, _headDim });
            
            // Reshape to (batch, nHead, seqLen, headDim)
            k = k.reshape(batchSize, _nHead, seqLength, _headDim);
            v = v.reshape(batchSize, _nHead, seqLength, _headDim);
        }
        
        // Compute attention scores
        // (batch, nHead, seqLen, headDim) @ (batch, nHead, headDim, seqLen)
        // -> (batch, nHead, seqLen, seqLen)
        var scores = matmul(q, k.transpose(-2, -1)) * _scale;
        
        // Apply causal mask: prevent attending to future positions
        // Create a mask where mask[i, j] = -inf if i < j (position i cannot attend to position j > i)
        var causalMask = ones(seqLength, seqLength, dtype: x.dtype, device: x.device)
            .tril()  // Lower triangular matrix (1s on and below diagonal)
            .log();  // Convert 1 -> 0, 0 -> -inf
        
        // Apply sliding window mask if specified
        if (_windowSize.HasValue)
        {
            // Create a band matrix: only allow attention within window
            var windowMask = ones(seqLength, seqLength, dtype: x.dtype, device: x.device)
                .triu(-_windowSize.Value)  // Upper triangular with offset
                .tril()  // Lower triangular
                .log();
            
            // Combine causal and window masks (take maximum to keep valid positions)
            causalMask = maximum(causalMask, windowMask);
        }
        
        // Add mask to scores (broadcasting over batch and head dimensions)
        scores = scores + causalMask;
        
        // Apply softmax to get attention weights
        // (batch, nHead, seqLen, seqLen)
        var attnWeights = softmax(scores, dim: -1);
        
        // Apply attention weights to values
        // (batch, nHead, seqLen, seqLen) @ (batch, nHead, seqLen, headDim)
        // -> (batch, nHead, seqLen, headDim)
        var output = matmul(attnWeights, v);
        
        // Transpose back and reshape
        // (batch, nHead, seqLen, headDim) -> (batch, seqLen, nHead, headDim)
        output = output.transpose(1, 2);
        
        // Concatenate heads
        // (batch, seqLen, nHead, headDim) -> (batch, seqLen, nHead * headDim)
        output = output.contiguous().view(batchSize, seqLength, _nHead * _headDim);
        
        // Final output projection
        output = _outProj.forward(output);
        
        return output;
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
        }
        base.Dispose(disposing);
    }
}
