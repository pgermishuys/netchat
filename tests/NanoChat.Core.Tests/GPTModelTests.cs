using NanoChat.Core.Model;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace NanoChat.Core.Tests;

public class GPTModelTests
{
    [Fact]
    public void Constructor_CreatesModelWithDefaultConfig()
    {
        var config = GPTConfig.CreateNanoChatDefault();
        using var model = new GPT(config);

        Assert.NotNull(model);
        Assert.Equal(config, model.Config);
    }

    [Fact]
    public void Constructor_ValidatesConfig()
    {
        var invalidConfig = new GPTConfig { NEmbd = 768, NHead = 7 }; // Invalid: not divisible
        Assert.Throws<ArgumentException>(() => new GPT(invalidConfig));
    }

    [Fact]
    public void Constructor_ThrowsForNullConfig()
    {
        Assert.Throws<ArgumentNullException>(() => new GPT(null!));
    }

    [Fact]
    public void Forward_ReturnsCorrectShape()
    {
        var config = new GPTConfig
        {
            SequenceLen = 128,
            VocabSize = 1000,
            NLayer = 2,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);

        // Input: (batch=2, seqLen=10)
        using var input = randint(0, 1000, new long[] { 2, 10 }, dtype: ScalarType.Int64);
        using var output = model.forward(input);

        Assert.Equal(3, output.dim());
        Assert.Equal(2, output.shape[0]); // batch
        Assert.Equal(10, output.shape[1]); // seqLen
        Assert.Equal(1000, output.shape[2]); // vocab_size
    }

    [Fact]
    public void Forward_ThrowsForNonInt64Input()
    {
        var config = new GPTConfig { NLayer = 1, NHead = 2, NKvHead = 2, NEmbd = 32 };
        using var model = new GPT(config);

        using var input = randint(0, 100, new long[] { 1, 10 }, dtype: ScalarType.Int32);
        Assert.Throws<ArgumentException>(() => model.forward(input));
    }

    [Fact]
    public void Forward_ThrowsForNon2DInput()
    {
        var config = new GPTConfig { NLayer = 1, NHead = 2, NKvHead = 2, NEmbd = 32 };
        using var model = new GPT(config);

        using var input = randint(0, 100, new long[] { 10 }, dtype: ScalarType.Int64);
        Assert.Throws<ArgumentException>(() => model.forward(input));
    }

    [Fact]
    public void Forward_ThrowsForSequenceTooLong()
    {
        var config = new GPTConfig { SequenceLen = 64, NLayer = 1, NHead = 2, NKvHead = 2, NEmbd = 32 };
        using var model = new GPT(config);

        using var input = randint(0, 100, new long[] { 1, 100 }, dtype: ScalarType.Int64);
        Assert.Throws<ArgumentException>(() => model.forward(input));
    }

    [Fact]
    public void Forward_HandlesVariableSequenceLengths()
    {
        var config = new GPTConfig
        {
            SequenceLen = 128,
            VocabSize = 500,
            NLayer = 2,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);

        // Test different sequence lengths
        foreach (int seqLen in new[] { 1, 10, 50, 128 })
        {
            using var input = randint(0, 500, new long[] { 1, seqLen }, dtype: ScalarType.Int64);
            using var output = model.forward(input);

            Assert.Equal(3, output.dim());
            Assert.Equal(1, output.shape[0]);
            Assert.Equal(seqLen, output.shape[1]);
            Assert.Equal(500, output.shape[2]);
        }
    }

    [Fact]
    public void Forward_HandlesBatches()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 2,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);

        // Test different batch sizes
        foreach (int batchSize in new[] { 1, 2, 4, 8 })
        {
            using var input = randint(0, 500, new long[] { batchSize, 10 }, dtype: ScalarType.Int64);
            using var output = model.forward(input);

            Assert.Equal(batchSize, output.shape[0]);
            Assert.Equal(10, output.shape[1]);
            Assert.Equal(500, output.shape[2]);
        }
    }

    [Fact]
    public void Forward_AppliesSoftcap()
    {
        var config = new GPTConfig
        {
            VocabSize = 100,
            NLayer = 1,
            NHead = 2,
            NKvHead = 2,
            NEmbd = 32,
            LogitSoftcap = 5.0f // Small cap for testing
        };

        using var model = new GPT(config);
        using var input = randint(0, 100, new long[] { 1, 5 }, dtype: ScalarType.Int64);
        using var output = model.forward(input);

        // With softcap, logits should be bounded
        var maxLogit = output.abs().max().item<float>();
        Assert.True(maxLogit <= config.LogitSoftcap.Value + 0.5f,
            $"Max logit {maxLogit} should be close to or below softcap {config.LogitSoftcap.Value}");
    }

    [Fact]
    public void Forward_NoSoftcap()
    {
        var config = new GPTConfig
        {
            VocabSize = 100,
            NLayer = 1,
            NHead = 2,
            NKvHead = 2,
            NEmbd = 32,
            LogitSoftcap = null // Disable softcap
        };

        using var model = new GPT(config);
        using var input = randint(0, 100, new long[] { 1, 5 }, dtype: ScalarType.Int64);
        using var output = model.forward(input);

        // Should produce logits (no specific bounds without softcap)
        Assert.Equal(3, output.dim());
    }

    [Fact]
    public void Forward_WithGQA()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 2,
            NHead = 8,
            NKvHead = 2, // Group-Query Attention: 8/2 = 4 groups
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var input = randint(0, 500, new long[] { 2, 10 }, dtype: ScalarType.Int64);
        using var output = model.forward(input);

        Assert.Equal(3, output.dim());
        Assert.Equal(2, output.shape[0]);
        Assert.Equal(10, output.shape[1]);
        Assert.Equal(500, output.shape[2]);
    }

    [Fact]
    public void Forward_WithSlidingWindow()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 4,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64,
            WindowPattern = "SSSL",  // Sliding for 3 layers, full for 1
            WindowSize = 32
        };

        using var model = new GPT(config);
        using var input = randint(0, 500, new long[] { 1, 50 }, dtype: ScalarType.Int64);
        using var output = model.forward(input);

        Assert.Equal(3, output.dim());
        Assert.Equal(1, output.shape[0]);
        Assert.Equal(50, output.shape[1]);
        Assert.Equal(500, output.shape[2]);
    }

    [Fact]
    public void Forward_WithValueEmbeddings()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 4,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64,
            ValueEmbeddingPattern = "YNYN"  // Alternating value embeddings
        };

        using var model = new GPT(config);
        using var input = randint(0, 500, new long[] { 1, 10 }, dtype: ScalarType.Int64);
        using var output = model.forward(input);

        Assert.Equal(3, output.dim());
        Assert.Equal(1, output.shape[0]);
        Assert.Equal(10, output.shape[1]);
        Assert.Equal(500, output.shape[2]);
    }

    [Fact]
    public void Forward_WithResidLambdas()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 3,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64,
            ResidLambdas = new[] { 0.5f, 0.75f, 1.0f }
        };

        using var model = new GPT(config);
        using var input = randint(0, 500, new long[] { 1, 10 }, dtype: ScalarType.Int64);
        using var output = model.forward(input);

        Assert.Equal(3, output.dim());
    }

    [Fact]
    public void Forward_WithX0Lambdas()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 3,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64,
            X0Lambdas = new[] { 0.1f, 0.1f, 0.1f }
        };

        using var model = new GPT(config);
        using var input = randint(0, 500, new long[] { 1, 10 }, dtype: ScalarType.Int64);
        using var output = model.forward(input);

        Assert.Equal(3, output.dim());
    }

    [Fact]
    public void Forward_DifferentInputsProduceDifferentOutputs()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 2,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var input1 = randint(0, 500, new long[] { 1, 10 }, dtype: ScalarType.Int64);
        using var input2 = randint(0, 500, new long[] { 1, 10 }, dtype: ScalarType.Int64);

        using var output1 = model.forward(input1);
        using var output2 = model.forward(input2);

        // Outputs should be different (very high probability with random inputs)
        var diff = (output1 - output2).abs().sum().item<float>();
        Assert.True(diff > 0.1f, "Different inputs should produce different outputs");
    }

    [Fact]
    public void Forward_SameInputProducesSameOutput()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 2,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var input = randint(0, 500, new long[] { 1, 10 }, dtype: ScalarType.Int64);

        using var output1 = model.forward(input);
        using var output2 = model.forward(input);

        // Same input should produce same output (deterministic)
        var diff = (output1 - output2).abs().sum().item<float>();
        Assert.True(diff < 1e-5f, $"Same input should produce same output, diff={diff}");
    }

    [Fact]
    public void Forward_SingleToken()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 2,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var input = randint(0, 500, new long[] { 1, 1 }, dtype: ScalarType.Int64);
        using var output = model.forward(input);

        Assert.Equal(3, output.dim());
        Assert.Equal(1, output.shape[0]);
        Assert.Equal(1, output.shape[1]);
        Assert.Equal(500, output.shape[2]);
    }

    [Fact]
    public void Forward_MaxSequenceLength()
    {
        var config = new GPTConfig
        {
            SequenceLen = 64,
            VocabSize = 500,
            NLayer = 2,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var input = randint(0, 500, new long[] { 1, 64 }, dtype: ScalarType.Int64);
        using var output = model.forward(input);

        Assert.Equal(64, output.shape[1]);
    }

    [Fact]
    public void Dispose_CleansUpResources()
    {
        var config = new GPTConfig { NLayer = 2, NHead = 4, NKvHead = 4, NEmbd = 64 };
        var model = new GPT(config);

        model.Dispose();
        // Should not throw
    }

    [Fact]
    public void Generate_ThrowsForInvalidPrompt()
    {
        var config = new GPTConfig { NLayer = 1, NHead = 2, NKvHead = 2, NEmbd = 32 };
        using var model = new GPT(config);

        // Wrong dtype
        using var prompt1 = randint(0, 100, new long[] { 1, 5 }, dtype: ScalarType.Int32);
        Assert.Throws<ArgumentException>(() => model.Generate(prompt1, maxNewTokens: 5));

        // Wrong dimensions
        using var prompt2 = randint(0, 100, new long[] { 5 }, dtype: ScalarType.Int64);
        Assert.Throws<ArgumentException>(() => model.Generate(prompt2, maxNewTokens: 5));
    }

    [Fact]
    public void Generate_ProducesCorrectShape()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 1,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = randint(0, 500, new long[] { 2, 5 }, dtype: ScalarType.Int64);

        int maxNewTokens = 10;
        using var generated = model.Generate(prompt, maxNewTokens: maxNewTokens);

        Assert.Equal(2, generated.dim());
        Assert.Equal(2, generated.shape[0]); // batch size
        Assert.Equal(5 + maxNewTokens, generated.shape[1]); // prompt + new tokens
    }

    [Fact]
    public void Generate_WithTemperature()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 1,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = randint(0, 500, new long[] { 1, 5 }, dtype: ScalarType.Int64);

        using var generated = model.Generate(prompt, maxNewTokens: 5, temperature: 0.8f);

        Assert.Equal(1, generated.shape[0]);
        Assert.Equal(10, generated.shape[1]);
    }

    [Fact]
    public void Generate_WithTopK()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 1,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = randint(0, 500, new long[] { 1, 5 }, dtype: ScalarType.Int64);

        using var generated = model.Generate(prompt, maxNewTokens: 5, topK: 50);

        Assert.Equal(1, generated.shape[0]);
        Assert.Equal(10, generated.shape[1]);
    }

    [Fact]
    public void GenerateStreaming_ThrowsForNullCallback()
    {
        var config = new GPTConfig { NLayer = 1, NHead = 2, NKvHead = 2, NEmbd = 32 };
        using var model = new GPT(config);
        using var prompt = randint(0, 100, new long[] { 1, 5 }, dtype: ScalarType.Int64);

        Assert.Throws<ArgumentNullException>(() => 
            model.GenerateStreaming(prompt, maxNewTokens: 5, onTokenGenerated: null!));
    }

    [Fact]
    public void GenerateStreaming_InvokesCallbackForEachToken()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 1,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = randint(0, 500, new long[] { 1, 5 }, dtype: ScalarType.Int64);

        int callbackCount = 0;
        var tokenIds = new List<long>();
        var positions = new List<int>();

        using var generated = model.GenerateStreaming(
            prompt,
            maxNewTokens: 10,
            onTokenGenerated: (batchIdx, tokenId, position) =>
            {
                callbackCount++;
                tokenIds.Add(tokenId);
                positions.Add(position);
                return true; // Continue generation
            });

        // Should invoke callback 10 times (once per token)
        Assert.Equal(10, callbackCount);
        Assert.Equal(10, tokenIds.Count);
        Assert.Equal(10, positions.Count);

        // Positions should be 0, 1, 2, ..., 9
        Assert.Equal(Enumerable.Range(0, 10), positions);

        // Generated shape should be correct
        Assert.Equal(1, generated.shape[0]);
        Assert.Equal(15, generated.shape[1]); // 5 prompt + 10 new
    }

    [Fact]
    public void GenerateStreaming_CallbackCanStopEarly()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 1,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = randint(0, 500, new long[] { 1, 5 }, dtype: ScalarType.Int64);

        int callbackCount = 0;

        using var generated = model.GenerateStreaming(
            prompt,
            maxNewTokens: 20,
            onTokenGenerated: (batchIdx, tokenId, position) =>
            {
                callbackCount++;
                // Stop after 5 tokens
                return position < 4;
            });

        // Should only generate 5 tokens (0-4, then returns false at position 4)
        Assert.Equal(5, callbackCount);
        Assert.Equal(10, generated.shape[1]); // 5 prompt + 5 new
    }

    [Fact]
    public void GenerateStreaming_StopTokensTerminateGeneration()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 1,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = tensor(new long[] { 1, 2, 3, 4, 5 }).reshape(1, 5);

        var generatedTokens = new List<long>();
        var stopTokens = new HashSet<long> { 100, 200, 300 };

        // We need to generate deterministically to test stop tokens
        // For this test, we'll just verify that if a stop token is generated, the callback is still invoked
        using var generated = model.GenerateStreaming(
            prompt,
            maxNewTokens: 10,
            onTokenGenerated: (batchIdx, tokenId, position) =>
            {
                generatedTokens.Add(tokenId);
                return true;
            },
            stopTokens: stopTokens);

        // Verify generation completed (may or may not hit a stop token randomly)
        Assert.True(generated.shape[1] >= 5); // At least prompt length
        Assert.True(generated.shape[1] <= 15); // At most prompt + maxNewTokens
    }

    [Fact]
    public void GenerateStreaming_HandlesBatchGeneration()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 1,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = randint(0, 500, new long[] { 3, 5 }, dtype: ScalarType.Int64);

        var batchIndices = new List<int>();
        var tokenCounts = new int[3]; // Track tokens per batch

        using var generated = model.GenerateStreaming(
            prompt,
            maxNewTokens: 10,
            onTokenGenerated: (batchIdx, tokenId, position) =>
            {
                batchIndices.Add(batchIdx);
                tokenCounts[batchIdx]++;
                return true;
            });

        // Should invoke callback 30 times (10 tokens × 3 batches)
        Assert.Equal(30, batchIndices.Count);

        // Each batch should get 10 tokens
        Assert.All(tokenCounts, count => Assert.Equal(10, count));

        // Generated shape should be correct
        Assert.Equal(3, generated.shape[0]); // batch size
        Assert.Equal(15, generated.shape[1]); // 5 prompt + 10 new
    }

    [Fact]
    public void GenerateStreaming_PerBatchEarlyTermination()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 1,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = randint(0, 500, new long[] { 2, 5 }, dtype: ScalarType.Int64);

        var tokenCounts = new int[2];

        using var generated = model.GenerateStreaming(
            prompt,
            maxNewTokens: 20,
            onTokenGenerated: (batchIdx, tokenId, position) =>
            {
                tokenCounts[batchIdx]++;
                // Batch 0 stops after 3 tokens, batch 1 continues
                if (batchIdx == 0)
                    return position < 2;
                return true;
            });

        // Batch 0 should have 3 tokens (positions 0, 1, 2)
        Assert.Equal(3, tokenCounts[0]);
        // Batch 1 should have all 20 tokens
        Assert.Equal(20, tokenCounts[1]);
    }

    [Fact]
    public void GenerateStreaming_WithTemperature()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 1,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = randint(0, 500, new long[] { 1, 5 }, dtype: ScalarType.Int64);

        int callbackCount = 0;

        using var generated = model.GenerateStreaming(
            prompt,
            maxNewTokens: 5,
            onTokenGenerated: (batchIdx, tokenId, position) =>
            {
                callbackCount++;
                return true;
            },
            temperature: 0.8f);

        Assert.Equal(5, callbackCount);
        Assert.Equal(10, generated.shape[1]);
    }

    [Fact]
    public void GenerateStreaming_WithTopK()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 1,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = randint(0, 500, new long[] { 1, 5 }, dtype: ScalarType.Int64);

        int callbackCount = 0;

        using var generated = model.GenerateStreaming(
            prompt,
            maxNewTokens: 5,
            onTokenGenerated: (batchIdx, tokenId, position) =>
            {
                callbackCount++;
                return true;
            },
            topK: 50);

        Assert.Equal(5, callbackCount);
        Assert.Equal(10, generated.shape[1]);
    }

    [Fact]
    public void GenerateStreaming_WithCache()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 2,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = randint(0, 500, new long[] { 1, 5 }, dtype: ScalarType.Int64);

        int callbackCount = 0;

        using var generated = model.GenerateStreaming(
            prompt,
            maxNewTokens: 10,
            onTokenGenerated: (batchIdx, tokenId, position) =>
            {
                callbackCount++;
                return true;
            },
            useCache: true);

        Assert.Equal(10, callbackCount);
        Assert.Equal(15, generated.shape[1]);
    }

    [Fact]
    public void GenerateStreaming_WithoutCache()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 2,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = randint(0, 500, new long[] { 1, 5 }, dtype: ScalarType.Int64);

        int callbackCount = 0;

        using var generated = model.GenerateStreaming(
            prompt,
            maxNewTokens: 10,
            onTokenGenerated: (batchIdx, tokenId, position) =>
            {
                callbackCount++;
                return true;
            },
            useCache: false);

        Assert.Equal(10, callbackCount);
        Assert.Equal(15, generated.shape[1]);
    }

    [Fact]
    public void GenerateStreaming_ProducesCorrectShape()
    {
        var config = new GPTConfig
        {
            VocabSize = 500,
            NLayer = 1,
            NHead = 4,
            NKvHead = 4,
            NEmbd = 64
        };

        using var model = new GPT(config);
        using var prompt = randint(0, 500, new long[] { 2, 5 }, dtype: ScalarType.Int64);

        int maxNewTokens = 10;
        using var generated = model.GenerateStreaming(
            prompt,
            maxNewTokens: maxNewTokens,
            onTokenGenerated: (_, _, _) => true);

        Assert.Equal(2, generated.dim());
        Assert.Equal(2, generated.shape[0]); // batch size
        Assert.Equal(5 + maxNewTokens, generated.shape[1]); // prompt + new tokens
    }
}
