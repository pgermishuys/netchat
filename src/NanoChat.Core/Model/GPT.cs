using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NanoChat.Core.Model;

/// <summary>
/// GPT language model with configurable transformer architecture.
/// Implements the nanochat model with support for:
/// - Token embeddings
/// - Multiple transformer blocks
/// - RMSNorm
/// - Rotary position embeddings
/// - Group-Query Attention (GQA)
/// - Sliding window attention
/// - Value embeddings (ResFormer-style)
/// - Logit softcapping
/// </summary>
public class GPT : Module<Tensor, Tensor>
{
    private readonly GPTConfig _config;
    private readonly Module<Tensor, Tensor> _tokenEmbedding;
    private readonly TransformerBlock[] _blocks;
    private readonly RMSNorm _finalNorm;
    private readonly Module<Tensor, Tensor> _lmHead;

    /// <summary>
    /// Gets the model configuration.
    /// </summary>
    public GPTConfig Config => _config;

    /// <summary>
    /// Creates a new GPT model with the specified configuration.
    /// </summary>
    /// <param name="config">Model configuration.</param>
    public GPT(GPTConfig config) : base("GPT")
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _config.Validate();

        // Token embeddings (vocab_size -> n_embd)
        _tokenEmbedding = Embedding(_config.VocabSize, _config.NEmbd);

        // Transformer blocks
        _blocks = new TransformerBlock[_config.NLayer];
        for (int i = 0; i < _config.NLayer; i++)
        {
            _blocks[i] = new TransformerBlock(
                nEmbd: _config.NEmbd,
                nHead: _config.NHead,
                nKvHead: _config.NKvHead,
                maxSeqLen: _config.SequenceLen,
                hiddenDim: _config.GetMlpHiddenDim(),
                residLambda: _config.GetResidLambda(i),
                x0Lambda: _config.GetX0Lambda(i),
                windowSize: _config.GetWindowSize(i),
                useValueEmbedding: _config.GetUseValueEmbedding(i),
                ropeBase: _config.RopeBase
            );
            register_module($"block_{i}", _blocks[i]);
        }

        // Final normalization
        _finalNorm = new RMSNorm(eps: _config.RmsNormEps);

        // Language model head (n_embd -> vocab_size)
        // Note: No bias term
        _lmHead = Linear(_config.NEmbd, _config.VocabSize, hasBias: false);

        // Register all modules
        RegisterComponents();
    }

    /// <summary>
    /// Forward pass: input token IDs -> logits.
    /// </summary>
    /// <param name="input">Token IDs tensor of shape (batch, seqLen) with dtype int64.</param>
    /// <returns>Logits tensor of shape (batch, seqLen, vocabSize).</returns>
    public override Tensor forward(Tensor input)
    {
        if (input.dtype != ScalarType.Int64)
            throw new ArgumentException($"Input must be int64, got {input.dtype}");

        if (input.dim() != 2)
            throw new ArgumentException($"Input must be 2D (batch, seqLen), got shape {input.shape}");

        long batchSize = input.shape[0];
        long seqLen = input.shape[1];

        if (seqLen > _config.SequenceLen)
            throw new ArgumentException($"Sequence length {seqLen} exceeds max {_config.SequenceLen}");

        // Token embeddings: (batch, seqLen) -> (batch, seqLen, nEmbd)
        using var x = _tokenEmbedding.forward(input);

        // Keep reference to x0 for ResFormer-style residuals
        var x0 = x.alias();

        // Apply transformer blocks
        var current = x.alias();
        for (int i = 0; i < _config.NLayer; i++)
        {
            // Each block takes (x, seqLen, x0) and returns new x
            var next = _blocks[i].forward(current, seqLen, x0);
            current.Dispose();
            current = next;
        }

        // Final normalization
        using var normalized = _finalNorm.forward(current);
        current.Dispose();

        // Project to vocabulary: (batch, seqLen, nEmbd) -> (batch, seqLen, vocabSize)
        using var logits = _lmHead.forward(normalized);

        // Apply logit softcapping if configured
        Tensor finalLogits;
        if (_config.LogitSoftcap.HasValue)
        {
            float cap = _config.LogitSoftcap.Value;
            // softcap: cap * tanh(logits / cap)
            using var scaled = logits.div(cap);
            using var capped = scaled.tanh();
            finalLogits = capped.mul(cap);
        }
        else
        {
            finalLogits = logits.alias();
        }

        // Clean up x0
        x0.Dispose();

        return finalLogits;
    }

    /// <summary>
    /// Forward pass with optional KV caching for efficient autoregressive generation.
    /// </summary>
    /// <param name="input">Token IDs tensor of shape (batch, seqLen) with dtype int64.</param>
    /// <param name="cache">Optional KV cache for efficient autoregressive generation.</param>
    /// <returns>Logits tensor of shape (batch, seqLen, vocabSize).</returns>
    public Tensor ForwardWithCache(Tensor input, KVCache? cache = null)
    {
        if (input.dtype != ScalarType.Int64)
            throw new ArgumentException($"Input must be int64, got {input.dtype}");

        if (input.dim() != 2)
            throw new ArgumentException($"Input must be 2D (batch, seqLen), got shape {input.shape}");

        long batchSize = input.shape[0];
        long seqLen = input.shape[1];
        
        // Total sequence length = cached length + new tokens
        long totalSeqLen = (cache?.CacheLength ?? 0) + seqLen;

        if (totalSeqLen > _config.SequenceLen)
            throw new ArgumentException($"Total sequence length {totalSeqLen} exceeds max {_config.SequenceLen}");

        // Token embeddings: (batch, seqLen) -> (batch, seqLen, nEmbd)
        using var x = _tokenEmbedding.forward(input);

        // Keep reference to x0 for ResFormer-style residuals
        var x0 = x.alias();

        // Apply transformer blocks with caching
        var current = x.alias();
        for (int i = 0; i < _config.NLayer; i++)
        {
            // Each block takes (x, totalSeqLen, x0, cache, layerIdx) and returns new x
            var next = _blocks[i].ForwardWithCache(current, totalSeqLen, x0, cache, i);
            current.Dispose();
            current = next;
        }

        // Final normalization
        using var normalized = _finalNorm.forward(current);
        current.Dispose();

        // Project to vocabulary: (batch, seqLen, nEmbd) -> (batch, seqLen, vocabSize)
        using var logits = _lmHead.forward(normalized);

        // Apply logit softcapping if configured
        Tensor finalLogits;
        if (_config.LogitSoftcap.HasValue)
        {
            float cap = _config.LogitSoftcap.Value;
            // softcap: cap * tanh(logits / cap)
            using var scaled = logits.div(cap);
            using var capped = scaled.tanh();
            finalLogits = capped.mul(cap);
        }
        else
        {
            finalLogits = logits.alias();
        }

        // Clean up x0
        x0.Dispose();

        return finalLogits;
    }

    /// <summary>
    /// Generates text autoregressively from a prompt using KV caching for improved performance.
    /// </summary>
    /// <param name="prompt">Initial token IDs of shape (batch, promptLen).</param>
    /// <param name="maxNewTokens">Maximum number of new tokens to generate.</param>
    /// <param name="temperature">Sampling temperature (default: 1.0).</param>
    /// <param name="topK">Top-k sampling parameter (default: null, disabled).</param>
    /// <param name="useCache">Whether to use KV caching (default: true for better performance).</param>
    /// <returns>Generated token IDs of shape (batch, promptLen + maxNewTokens).</returns>
    public Tensor Generate(
        Tensor prompt,
        int maxNewTokens,
        float temperature = 1.0f,
        int? topK = null,
        bool useCache = true)
    {
        if (prompt.dtype != ScalarType.Int64)
            throw new ArgumentException($"Prompt must be int64, got {prompt.dtype}");

        if (prompt.dim() != 2)
            throw new ArgumentException($"Prompt must be 2D (batch, promptLen), got shape {prompt.shape}");

        long batchSize = prompt.shape[0];
        long promptLen = prompt.shape[1];

        // Start with the prompt
        var generated = prompt.clone();

        using (no_grad())
        {
            // Create KV cache if enabled
            KVCache? cache = null;
            if (useCache)
            {
                cache = new KVCache(
                    nLayers: _config.NLayer,
                    nHeads: _config.NHead,
                    headDim: _config.NEmbd / _config.NHead,
                    batchSize: (int)batchSize,
                    maxSeqLen: _config.SequenceLen
                );
            }

            try
            {
                for (int i = 0; i < maxNewTokens; i++)
                {
                    long currentLen = generated.shape[1];

                    // Prepare input for this step
                    Tensor input;
                    if (cache is not null && i > 0)
                    {
                        // With cache: only process the last token
                        input = generated[TensorIndex.Colon, TensorIndex.Single(-1)].unsqueeze(-1);
                    }
                    else if (currentLen > _config.SequenceLen)
                    {
                        // Without cache or first iteration: truncate to context window if needed
                        var startIdx = currentLen - _config.SequenceLen;
                        input = generated[TensorIndex.Ellipsis, TensorIndex.Slice(startIdx)].alias();
                        
                        // Clear cache if we had to truncate (context window exceeded)
                        if (cache is not null)
                        {
                            cache.Clear();
                        }
                    }
                    else
                    {
                        input = generated.alias();
                    }

                    // Forward pass with optional caching
                    using var logits = useCache ? ForwardWithCache(input, cache) : forward(input);
                    input.Dispose();

                // Get logits for last token: (batch, vocabSize)
                using var lastLogits = logits[TensorIndex.Colon, TensorIndex.Single(-1), TensorIndex.Colon];

                // Apply temperature
                using var scaledLogits = lastLogits.div(temperature);

                // Sample next token
                Tensor nextToken;
                if (topK.HasValue)
                {
                    // Top-k sampling (clamp k to vocab size)
                    int k = Math.Min(topK.Value, (int)_config.VocabSize);
                    var topKResult = scaledLogits.topk(k, dim: -1);
                    var topKValues = topKResult.values;
                    var topKIndices = topKResult.indexes;

                    // Apply softmax to top-k values
                    using var probs = functional.softmax(topKValues, dim: -1);

                    // Sample from top-k distribution
                    using var samples = probs.multinomial(1);

                    // Map back to original indices
                    nextToken = topKIndices.gather(dim: -1, index: samples);
                    
                    topKValues.Dispose();
                    topKIndices.Dispose();
                }
                else
                {
                    // Standard sampling
                    using var probs = functional.softmax(scaledLogits, dim: -1);
                    nextToken = probs.multinomial(1);
                }

                    // Append to generated sequence: (batch, seqLen) + (batch, 1) -> (batch, seqLen+1)
                    using var temp = generated;
                    generated = cat(new[] { temp, nextToken }, dim: -1);
                    nextToken.Dispose();
                }
            }
            finally
            {
                // Clean up cache
                cache?.Dispose();
            }
        }

        return generated;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _tokenEmbedding?.Dispose();
            if (_blocks != null)
            {
                foreach (var block in _blocks)
                {
                    block?.Dispose();
                }
            }
            _finalNorm?.Dispose();
            _lmHead?.Dispose();
        }

        base.Dispose(disposing);
    }
}
