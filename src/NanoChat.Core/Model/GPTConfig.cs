namespace NanoChat.Core.Model;

/// <summary>
/// Configuration for the GPT model architecture.
/// </summary>
public record GPTConfig
{
    /// <summary>
    /// Maximum sequence length.
    /// </summary>
    public int SequenceLen { get; init; } = 2048;

    /// <summary>
    /// Vocabulary size (padded to multiple of 64).
    /// </summary>
    public int VocabSize { get; init; } = 32768;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    public int NLayer { get; init; } = 12;

    /// <summary>
    /// Number of attention heads per layer.
    /// </summary>
    public int NHead { get; init; } = 6;

    /// <summary>
    /// Number of key-value heads (for Group-Query Attention).
    /// Must evenly divide NHead. Set to NHead for standard MHA.
    /// </summary>
    public int NKvHead { get; init; } = 6;

    /// <summary>
    /// Embedding dimension.
    /// </summary>
    public int NEmbd { get; init; } = 768;

    /// <summary>
    /// Sliding window pattern for attention layers.
    /// Each character represents one layer:
    /// - 'S': Sliding window attention
    /// - 'L': Full attention (entire sequence)
    /// Example: "SSSL" means sliding for layers 0-2, full for layer 3, and repeats.
    /// </summary>
    public string WindowPattern { get; init; } = "SSSL";

    /// <summary>
    /// Sliding window size for layers marked 'S' in WindowPattern.
    /// </summary>
    public int WindowSize { get; init; } = 512;

    /// <summary>
    /// MLP hidden dimension. If null, defaults to 4 * NEmbd.
    /// </summary>
    public int? MlpHiddenDim { get; init; } = null;

    /// <summary>
    /// Per-layer residual scaling factors (resid_lambda).
    /// If null, all layers use 1.0.
    /// </summary>
    public float[]? ResidLambdas { get; init; } = null;

    /// <summary>
    /// Per-layer ResFormer residual scaling factors (x0_lambda).
    /// If null, all layers use 0.0 (disabled).
    /// </summary>
    public float[]? X0Lambdas { get; init; } = null;

    /// <summary>
    /// Whether to use value embeddings in attention layers.
    /// Can be a pattern string like "YNYN" for alternating layers.
    /// If null or empty, no value embeddings are used.
    /// </summary>
    public string? ValueEmbeddingPattern { get; init; } = null;

    /// <summary>
    /// Epsilon for RMSNorm numerical stability.
    /// </summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>
    /// Base for rotary embeddings.
    /// </summary>
    public float RopeBase { get; init; } = 10000f;

    /// <summary>
    /// Logit softcap value. If null, no softcapping is applied.
    /// nanochat default: 15.0 (applies: 15 * tanh(logits / 15))
    /// </summary>
    public float? LogitSoftcap { get; init; } = 15.0f;

    /// <summary>
    /// Dropout probability. Default is 0.0 (no dropout) for inference.
    /// </summary>
    public float Dropout { get; init; } = 0.0f;

    /// <summary>
    /// Validates the configuration and throws if invalid.
    /// </summary>
    public void Validate()
    {
        if (NEmbd <= 0)
            throw new ArgumentException($"NEmbd must be positive, got {NEmbd}");

        if (NHead <= 0)
            throw new ArgumentException($"NHead must be positive, got {NHead}");

        if (NKvHead <= 0)
            throw new ArgumentException($"NKvHead must be positive, got {NKvHead}");

        if (NEmbd % NHead != 0)
            throw new ArgumentException($"NEmbd ({NEmbd}) must be divisible by NHead ({NHead})");

        if (NHead % NKvHead != 0)
            throw new ArgumentException($"NHead ({NHead}) must be divisible by NKvHead ({NKvHead})");

        if (NLayer <= 0)
            throw new ArgumentException($"NLayer must be positive, got {NLayer}");

        if (VocabSize <= 0)
            throw new ArgumentException($"VocabSize must be positive, got {VocabSize}");

        if (SequenceLen <= 0)
            throw new ArgumentException($"SequenceLen must be positive, got {SequenceLen}");

        if (string.IsNullOrEmpty(WindowPattern))
            throw new ArgumentException("WindowPattern cannot be null or empty");

        if (WindowSize <= 0)
            throw new ArgumentException($"WindowSize must be positive, got {WindowSize}");

        if (ResidLambdas != null && ResidLambdas.Length != NLayer)
            throw new ArgumentException($"ResidLambdas length ({ResidLambdas.Length}) must match NLayer ({NLayer})");

        if (X0Lambdas != null && X0Lambdas.Length != NLayer)
            throw new ArgumentException($"X0Lambdas length ({X0Lambdas.Length}) must match NLayer ({NLayer})");

        // Validate window pattern contains only 'S' and 'L'
        foreach (char c in WindowPattern)
        {
            if (c != 'S' && c != 'L')
                throw new ArgumentException($"WindowPattern must contain only 'S' or 'L', found '{c}'");
        }

        // Validate value embedding pattern if present
        if (!string.IsNullOrEmpty(ValueEmbeddingPattern))
        {
            foreach (char c in ValueEmbeddingPattern)
            {
                if (c != 'Y' && c != 'N')
                    throw new ArgumentException($"ValueEmbeddingPattern must contain only 'Y' or 'N', found '{c}'");
            }
        }
    }

    /// <summary>
    /// Gets the head dimension (embedding dimension per head).
    /// </summary>
    public int HeadDim => NEmbd / NHead;

    /// <summary>
    /// Gets the MLP hidden dimension (defaults to 4 * NEmbd if not specified).
    /// </summary>
    public int GetMlpHiddenDim() => MlpHiddenDim ?? (4 * NEmbd);

    /// <summary>
    /// Gets the residual lambda for a specific layer (defaults to 1.0).
    /// </summary>
    public float GetResidLambda(int layer)
    {
        if (layer < 0 || layer >= NLayer)
            throw new ArgumentOutOfRangeException(nameof(layer), $"Layer must be in [0, {NLayer})");

        return ResidLambdas?[layer] ?? 1.0f;
    }

    /// <summary>
    /// Gets the x0 lambda for a specific layer (defaults to 0.0).
    /// </summary>
    public float GetX0Lambda(int layer)
    {
        if (layer < 0 || layer >= NLayer)
            throw new ArgumentOutOfRangeException(nameof(layer), $"Layer must be in [0, {NLayer})");

        return X0Lambdas?[layer] ?? 0.0f;
    }

    /// <summary>
    /// Gets the sliding window size for a specific layer based on WindowPattern.
    /// Returns null for full attention ('L'), or WindowSize for sliding attention ('S').
    /// </summary>
    public int? GetWindowSize(int layer)
    {
        if (layer < 0 || layer >= NLayer)
            throw new ArgumentOutOfRangeException(nameof(layer), $"Layer must be in [0, {NLayer})");

        // Pattern repeats cyclically
        char patternChar = WindowPattern[layer % WindowPattern.Length];
        return patternChar == 'S' ? WindowSize : null;
    }

    /// <summary>
    /// Gets whether value embeddings should be used for a specific layer.
    /// </summary>
    public bool GetUseValueEmbedding(int layer)
    {
        if (layer < 0 || layer >= NLayer)
            throw new ArgumentOutOfRangeException(nameof(layer), $"Layer must be in [0, {NLayer})");

        if (string.IsNullOrEmpty(ValueEmbeddingPattern))
            return false;

        // Pattern repeats cyclically
        char patternChar = ValueEmbeddingPattern[layer % ValueEmbeddingPattern.Length];
        return patternChar == 'Y';
    }

    /// <summary>
    /// Creates a default nanochat configuration.
    /// </summary>
    public static GPTConfig CreateNanoChatDefault() => new()
    {
        SequenceLen = 2048,
        VocabSize = 32768,
        NLayer = 12,
        NHead = 6,
        NKvHead = 6,
        NEmbd = 768,
        WindowPattern = "SSSL",
        WindowSize = 512,
        MlpHiddenDim = null, // Will default to 4 * 768 = 3072
        ResidLambdas = null,
        X0Lambdas = null,
        ValueEmbeddingPattern = null,
        RmsNormEps = 1e-6f,
        RopeBase = 10000f,
        LogitSoftcap = 15.0f,
        Dropout = 0.0f
    };
}
