using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NanoChat.Core.Model;

/// <summary>
/// Transformer Block with pre-normalization and residual connections
/// 
/// Implements a standard transformer block with the following architecture:
/// - Pre-normalization using RMSNorm before each sub-layer
/// - Multi-head causal self-attention with optional GQA support
/// - Feed-forward MLP network with ReLU² activation
/// - Residual connections with per-layer scaling factors
/// 
/// Architecture:
/// x1 = x + residLambda * attention(norm(x))
/// x2 = x1 + residLambda * mlp(norm(x1))
/// 
/// Optional ResFormer-style residuals with x0_lambda:
/// x1 = x + residLambda * attention(norm(x)) + x0Lambda * x0
/// x2 = x1 + residLambda * mlp(norm(x1)) + x0Lambda * x0
/// 
/// Reference: "Attention Is All You Need" + nanochat-specific modifications
/// </summary>
public class TransformerBlock : Module<Tensor, long, Tensor?, Tensor>
{
    private readonly RMSNorm _attnNorm;
    private readonly Attention _attn;
    private readonly RMSNorm _mlpNorm;
    private readonly MLP _mlp;
    private readonly float _residLambda;
    private readonly float _x0Lambda;

    /// <summary>
    /// Initialize Transformer Block
    /// </summary>
    /// <param name="nEmbd">Embedding dimension</param>
    /// <param name="nHead">Number of attention heads</param>
    /// <param name="nKvHead">Number of key-value heads (for GQA). If null, defaults to nHead</param>
    /// <param name="windowSize">Sliding window size for attention. If null, use full attention</param>
    /// <param name="hiddenDim">MLP hidden dimension. If null, defaults to 4 * nEmbd</param>
    /// <param name="residLambda">Residual connection scaling factor</param>
    /// <param name="x0Lambda">Initial input (x0) scaling factor for ResFormer-style residuals</param>
    /// <param name="maxSeqLen">Maximum sequence length for RoPE</param>
    /// <param name="ropeBase">Base for RoPE frequency computation</param>
    /// <param name="useValueEmbedding">Whether to use ResFormer-style value embeddings in this layer</param>
    /// <param name="name">Module name</param>
    public TransformerBlock(
        int nEmbd,
        int nHead,
        int? nKvHead = null,
        int? windowSize = null,
        int? hiddenDim = null,
        float residLambda = 1.0f,
        float x0Lambda = 0.0f,
        int maxSeqLen = 2048,
        double ropeBase = 10000.0,
        bool useValueEmbedding = false,
        string? name = null)
        : base(name ?? "TransformerBlock")
    {
        _residLambda = residLambda;
        _x0Lambda = x0Lambda;

        // Pre-normalization for attention
        _attnNorm = new RMSNorm();

        // Multi-head attention with optional GQA
        _attn = new Attention(
            nEmbd: nEmbd,
            nHead: nHead,
            nKvHead: nKvHead,
            windowSize: windowSize,
            maxSeqLen: maxSeqLen,
            ropeBase: ropeBase,
            useValueEmbedding: useValueEmbedding);

        // Pre-normalization for MLP
        _mlpNorm = new RMSNorm();

        // Feed-forward MLP network
        _mlp = new MLP(nEmbd: nEmbd, hiddenDim: hiddenDim);

        // Register all modules for proper parameter tracking
        RegisterComponents();
    }

    /// <summary>
    /// Apply transformer block to input tensor
    /// 
    /// Architecture:
    /// 1. Attention sub-layer with residual:
    ///    x1 = x + residLambda * attention(norm(x)) + x0Lambda * x0
    /// 
    /// 2. MLP sub-layer with residual:
    ///    x2 = x1 + residLambda * mlp(norm(x1)) + x0Lambda * x0
    /// </summary>
    /// <param name="x">Input tensor of shape (batch, seqLen, nEmbd)</param>
    /// <param name="seqLen">Sequence length (used for RoPE)</param>
    /// <param name="x0">Initial input x0 for ResFormer-style residuals. If null, x0_lambda is ignored</param>
    /// <returns>Output tensor of shape (batch, seqLen, nEmbd)</returns>
    public override Tensor forward(Tensor x, long seqLen, Tensor? x0 = null)
    {
        return ForwardWithCache(x, seqLen, x0, null, -1);
    }
    
    /// <summary>
    /// Apply transformer block to input tensor with optional KV caching
    /// </summary>
    /// <param name="x">Input tensor of shape (batch, seqLen, nEmbd)</param>
    /// <param name="seqLen">Total sequence length including cached tokens (used for RoPE)</param>
    /// <param name="x0">Initial input x0 for ResFormer-style residuals. If null, x0_lambda is ignored</param>
    /// <param name="cache">Optional KV cache for efficient autoregressive generation</param>
    /// <param name="layerIdx">Layer index for cache lookup (required if cache is provided)</param>
    /// <returns>Output tensor of shape (batch, seqLen, nEmbd)</returns>
    public Tensor ForwardWithCache(
        Tensor x, 
        long seqLen, 
        Tensor? x0 = null, 
        KVCache? cache = null, 
        int layerIdx = -1)
    {
        // Attention sub-layer with pre-normalization and residual
        // x1 = x + residLambda * attention(norm(x))
        var attnInput = _attnNorm.forward(x);
        var (attnOutput, _, _) = _attn.ForwardWithCache(attnInput, seqLen, cache, layerIdx);
        var x1 = x + _residLambda * attnOutput;

        // Add ResFormer-style x0 residual if provided
        if (x0 is not null && _x0Lambda != 0.0f)
        {
            x1 = x1 + _x0Lambda * x0;
        }

        // MLP sub-layer with pre-normalization and residual
        // x2 = x1 + residLambda * mlp(norm(x1))
        var mlpInput = _mlpNorm.forward(x1);
        var mlpOutput = _mlp.forward(mlpInput);
        var x2 = x1 + _residLambda * mlpOutput;

        // Add ResFormer-style x0 residual if provided
        if (x0 is not null && _x0Lambda != 0.0f)
        {
            x2 = x2 + _x0Lambda * x0;
        }

        return x2;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _attnNorm?.Dispose();
            _attn?.Dispose();
            _mlpNorm?.Dispose();
            _mlp?.Dispose();
        }
        base.Dispose(disposing);
    }
}
