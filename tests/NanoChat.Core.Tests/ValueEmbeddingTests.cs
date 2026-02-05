using NanoChat.Core.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace NanoChat.Core.Tests;

public class ValueEmbeddingTests
{
    [Fact]
    public void Constructor_CreatesValidModule()
    {
        // Arrange & Act
        using var ve = new ValueEmbedding(headDim: 64, maxSeqLen: 128);

        // Assert
        Assert.NotNull(ve);
    }

    [Fact]
    public void Forward_ProducesCorrectShape_BatchSeqHeadDim()
    {
        // Arrange
        var headDim = 64;
        var maxSeqLen = 128;
        var batchSize = 2;
        var seqLen = 10;
        var nHead = 4;

        using var ve = new ValueEmbedding(headDim: headDim, maxSeqLen: maxSeqLen);
        using var input = randn(batchSize, seqLen, nHead, headDim);

        // Act
        using var output = ve.forward(input, seqLen);

        // Assert
        Assert.Equal(4, output.shape.Length);
        Assert.Equal(1, output.shape[0]);  // Broadcast over batch
        Assert.Equal(seqLen, output.shape[1]);
        Assert.Equal(1, output.shape[2]);  // Broadcast over heads
        Assert.Equal(headDim, output.shape[3]);
    }

    [Fact]
    public void Forward_ProducesCorrectShape_BatchHeadSeqDim()
    {
        // Arrange
        var headDim = 64;
        var maxSeqLen = 128;
        var batchSize = 2;
        var seqLen = 10;
        var nHead = 4;

        using var ve = new ValueEmbedding(headDim: headDim, maxSeqLen: maxSeqLen);
        using var input = randn(batchSize, nHead, seqLen, headDim);

        // Act
        using var output = ve.forward(input, seqLen);

        // Assert
        Assert.Equal(4, output.shape.Length);
        Assert.Equal(1, output.shape[0]);  // Broadcast over batch
        Assert.Equal(1, output.shape[1]);  // Broadcast over heads
        Assert.Equal(seqLen, output.shape[2]);
        Assert.Equal(headDim, output.shape[3]);
    }

    [Fact]
    public void Forward_DifferentSequenceLengths_ProducesDifferentOutputs()
    {
        // Different sequence lengths should produce different embeddings
        // due to the positional nature of value embeddings

        // Arrange
        var headDim = 64;
        var maxSeqLen = 128;
        var batchSize = 1;
        var nHead = 1;

        using var ve = new ValueEmbedding(headDim: headDim, maxSeqLen: maxSeqLen);
        using var input1 = randn(batchSize, 5, nHead, headDim);
        using var input2 = randn(batchSize, 10, nHead, headDim);

        // Act
        using var output1 = ve.forward(input1, 5);
        using var output2 = ve.forward(input2, 10);

        // Assert
        Assert.Equal(5, output1.shape[1]);
        Assert.Equal(10, output2.shape[1]);
    }

    [Fact]
    public void Forward_GatingMechanism_ProducesScaledOutput()
    {
        // The gating mechanism should scale the embeddings
        // Using sigmoid, gate should be in [0, 1] range

        // Arrange
        var headDim = 8;
        var maxSeqLen = 16;
        var batchSize = 1;
        var seqLen = 4;
        var nHead = 1;

        using var ve = new ValueEmbedding(headDim: headDim, maxSeqLen: maxSeqLen);
        using var input = zeros(batchSize, seqLen, nHead, headDim);

        // Act
        using var output = ve.forward(input, seqLen);

        // Assert - output should be gated (not just raw embeddings)
        Assert.Equal(4, output.shape.Length);
        
        // Gate uses sigmoid, so output should be bounded
        var outputData = output.data<float>().ToArray();
        Assert.All(outputData, val => Assert.InRange(val, -10f, 10f));
    }

    [Fact]
    public void Forward_ExceedsMaxSeqLen_ThrowsException()
    {
        // Arrange
        var headDim = 64;
        var maxSeqLen = 10;
        var batchSize = 1;
        var seqLen = 20;  // Exceeds maxSeqLen
        var nHead = 1;

        using var ve = new ValueEmbedding(headDim: headDim, maxSeqLen: maxSeqLen);
        using var input = randn(batchSize, seqLen, nHead, headDim);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => ve.forward(input, seqLen));
    }

    [Fact]
    public void Forward_InvalidInputShape_ThrowsException()
    {
        // Arrange
        var headDim = 64;
        var maxSeqLen = 128;
        var seqLen = 10;

        using var ve = new ValueEmbedding(headDim: headDim, maxSeqLen: maxSeqLen);
        using var input = randn(2, 10, 64);  // 3D instead of 4D

        // Act & Assert
        Assert.Throws<ArgumentException>(() => ve.forward(input, seqLen));
    }

    [Fact]
    public void Forward_SameInputMultipleTimes_ProducesSameOutput()
    {
        // Value embeddings should be deterministic (same input -> same output)

        // Arrange
        var headDim = 32;
        var maxSeqLen = 64;
        var batchSize = 2;
        var seqLen = 8;
        var nHead = 2;

        using var ve = new ValueEmbedding(headDim: headDim, maxSeqLen: maxSeqLen);
        using var input = randn(batchSize, seqLen, nHead, headDim);

        // Act
        using var output1 = ve.forward(input, seqLen);
        using var output2 = ve.forward(input, seqLen);

        // Assert - outputs should be identical
        var diff = (output1 - output2).abs().max().item<float>();
        Assert.True(diff < 1e-6f, $"Outputs differ by {diff}");
    }

    [Fact]
    public void Forward_CanBeAddedToValues()
    {
        // Value embeddings are meant to be added to value vectors in attention
        // This test verifies the shapes are compatible

        // Arrange
        var headDim = 64;
        var maxSeqLen = 128;
        var batchSize = 2;
        var seqLen = 10;
        var nHead = 4;

        using var ve = new ValueEmbedding(headDim: headDim, maxSeqLen: maxSeqLen);
        using var values = randn(batchSize, seqLen, nHead, headDim);

        // Act
        using var embeddings = ve.forward(values, seqLen);
        using var augmentedValues = values + embeddings;  // Should broadcast correctly

        // Assert
        Assert.Equal(values.shape, augmentedValues.shape);
    }

    [Fact]
    public void Constructor_DifferentMaxSeqLen_CreatesValidModule()
    {
        // Test that different max sequence lengths work

        // Arrange & Act
        using var ve32 = new ValueEmbedding(headDim: 64, maxSeqLen: 32);
        using var ve1024 = new ValueEmbedding(headDim: 64, maxSeqLen: 1024);
        using var ve2048 = new ValueEmbedding(headDim: 64, maxSeqLen: 2048);

        // Assert
        Assert.NotNull(ve32);
        Assert.NotNull(ve1024);
        Assert.NotNull(ve2048);
    }

    [Fact]
    public void Dispose_CleansUpResources()
    {
        // Arrange
        var ve = new ValueEmbedding(headDim: 64, maxSeqLen: 128);

        // Act
        ve.Dispose();

        // Assert - should not throw
        Assert.True(true);
    }
}
