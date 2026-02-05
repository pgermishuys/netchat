using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NanoChat.Core.Model;

/// <summary>
/// Value Embedding module with gating mechanism (ResFormer-style)
/// 
/// Value embeddings provide positional information directly to the value vectors
/// in attention, complementing rotary position embeddings on queries and keys.
/// 
/// The gating mechanism allows the model to learn how much to use the value embeddings:
/// ve_output = gate * value_embedding[position]
/// 
/// This is typically applied to alternating layers in the transformer stack.
/// 
/// Reference: ResFormer architecture
/// </summary>
public class ValueEmbedding : Module<Tensor, long, Tensor>
{
    private readonly int _maxSeqLen;
    private readonly int _headDim;

    /// <summary>
    /// Initialize Value Embedding module
    /// </summary>
    /// <param name="headDim">Dimension of each attention head</param>
    /// <param name="maxSeqLen">Maximum sequence length to support</param>
    /// <param name="name">Module name</param>
    public ValueEmbedding(
        int headDim,
        int maxSeqLen = 2048,
        string? name = null)
        : base(name ?? "ValueEmbedding")
    {
        _headDim = headDim;
        _maxSeqLen = maxSeqLen;

        // Create learnable value embeddings for each position
        // Shape: (maxSeqLen, headDim)
        var embeddings = randn(maxSeqLen, headDim) * 0.02f;
        embeddings.requires_grad = true;

        // Learnable gate parameter (scalar, initialized near 0 for stability)
        var gate = zeros(1);
        gate.requires_grad = true;

        // Register parameters for training
        register_parameter("embeddings", new TorchSharp.Modules.Parameter(embeddings));
        register_parameter("gate", new TorchSharp.Modules.Parameter(gate));
    }

    /// <summary>
    /// Apply gated value embeddings
    /// 
    /// For each position in the sequence, adds a gated value embedding:
    /// output[pos] = gate * embeddings[pos]
    /// 
    /// These are typically added to the value vectors in attention:
    /// v = v_proj(x) + value_embedding(seqLen)
    /// </summary>
    /// <param name="x">Input tensor of shape (batch, seqLen, nHead, headDim) or (batch, nHead, seqLen, headDim)</param>
    /// <param name="seqLen">Sequence length to use for embedding lookup</param>
    /// <returns>Value embeddings tensor with same shape as input</returns>
    public override Tensor forward(Tensor x, long seqLen)
    {
        if (seqLen > _maxSeqLen)
        {
            throw new ArgumentException(
                $"Sequence length {seqLen} exceeds maximum supported length {_maxSeqLen}");
        }

        // Retrieve parameters from the module
        var embeddings = get_parameter("embeddings");
        var gateParam = get_parameter("gate");

        if (embeddings is null)
        {
            throw new InvalidOperationException("Embeddings parameter not found");
        }

        if (gateParam is null)
        {
            throw new InvalidOperationException("Gate parameter not found");
        }

        // Get embeddings for the current sequence length
        // Shape: (seqLen, headDim)
        var posEmbeddings = embeddings[TensorIndex.Slice(0, seqLen)];

        // Apply gating: gate * embeddings
        // Using sigmoid to keep gate in [0, 1] range for stability
        var gate = sigmoid(gateParam);
        var gatedEmbeddings = gate * posEmbeddings;

        // Determine input shape and broadcast accordingly
        // Support both (batch, seqLen, nHead, headDim) and (batch, nHead, seqLen, headDim)
        var shape = x.shape;
        
        if (shape.Length == 4)
        {
            // Input is (batch, seqLen, nHead, headDim) or (batch, nHead, seqLen, headDim)
            // Check which dimension is seqLen
            if (shape[1] == seqLen)
            {
                // Shape: (batch, seqLen, nHead, headDim)
                // Broadcast embeddings: (seqLen, headDim) -> (1, seqLen, 1, headDim)
                gatedEmbeddings = gatedEmbeddings.unsqueeze(0).unsqueeze(2);
            }
            else if (shape[2] == seqLen)
            {
                // Shape: (batch, nHead, seqLen, headDim)
                // Broadcast embeddings: (seqLen, headDim) -> (1, 1, seqLen, headDim)
                gatedEmbeddings = gatedEmbeddings.unsqueeze(0).unsqueeze(0);
            }
            else
            {
                throw new ArgumentException(
                    $"Expected sequence length {seqLen} in dimension 1 or 2, but got shape {string.Join(", ", shape)}");
            }
        }
        else
        {
            throw new ArgumentException(
                $"Expected 4D tensor, but got {shape.Length}D tensor with shape {string.Join(", ", shape)}");
        }

        return gatedEmbeddings;
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
    }
}
