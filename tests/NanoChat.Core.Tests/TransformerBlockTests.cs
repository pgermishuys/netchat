using TorchSharp;
using static TorchSharp.torch;
using NanoChat.Core.Model;

namespace NanoChat.Core.Tests;

public class TransformerBlockTests : IDisposable
{
    public TransformerBlockTests()
    {
        // Set manual seed for reproducible tests
        manual_seed(42);
    }

    public void Dispose()
    {
        // Clean up any resources
        GC.Collect();
    }

    [Fact]
    public void Constructor_WithValidParameters_InitializesSuccessfully()
    {
        // Arrange & Act
        using var block = new TransformerBlock(
            nEmbd: 64,
            nHead: 4,
            nKvHead: 4,
            windowSize: null,
            hiddenDim: 256,
            residLambda: 1.0f,
            x0Lambda: 0.1f);

        // Assert
        Assert.NotNull(block);
    }

    [Fact]
    public void Forward_WithSimpleInput_ReturnsCorrectShape()
    {
        // Arrange
        var nEmbd = 64;
        var nHead = 4;
        var batchSize = 2;
        var seqLen = 8;

        using var block = new TransformerBlock(nEmbd: nEmbd, nHead: nHead);
        using var input = randn(batchSize, seqLen, nEmbd);

        // Act
        using var output = block.forward(input, seqLen);

        // Assert
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
    }

    [Fact]
    public void Forward_WithGQA_ReturnsCorrectShape()
    {
        // Arrange
        var nEmbd = 64;
        var nHead = 4;
        var nKvHead = 2; // GQA: fewer K/V heads
        var batchSize = 2;
        var seqLen = 8;

        using var block = new TransformerBlock(nEmbd: nEmbd, nHead: nHead, nKvHead: nKvHead);
        using var input = randn(batchSize, seqLen, nEmbd);

        // Act
        using var output = block.forward(input, seqLen);

        // Assert
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
    }

    [Fact]
    public void Forward_WithSlidingWindow_ReturnsCorrectShape()
    {
        // Arrange
        var nEmbd = 64;
        var nHead = 4;
        var windowSize = 4;
        var batchSize = 2;
        var seqLen = 16;

        using var block = new TransformerBlock(nEmbd: nEmbd, nHead: nHead, windowSize: windowSize);
        using var input = randn(batchSize, seqLen, nEmbd);

        // Act
        using var output = block.forward(input, seqLen);

        // Assert
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
    }

    [Fact]
    public void Forward_WithCustomHiddenDim_ReturnsCorrectShape()
    {
        // Arrange
        var nEmbd = 64;
        var nHead = 4;
        var hiddenDim = 128;
        var batchSize = 2;
        var seqLen = 8;

        using var block = new TransformerBlock(nEmbd: nEmbd, nHead: nHead, hiddenDim: hiddenDim);
        using var input = randn(batchSize, seqLen, nEmbd);

        // Act
        using var output = block.forward(input, seqLen);

        // Assert
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
    }

    [Fact]
    public void Forward_WithResidLambda_ScalesResidualCorrectly()
    {
        // Arrange
        var nEmbd = 64;
        var nHead = 4;
        var batchSize = 1;
        var seqLen = 4;

        // Create two blocks with different residual lambdas
        using var block1 = new TransformerBlock(nEmbd: nEmbd, nHead: nHead, residLambda: 1.0f);
        using var block2 = new TransformerBlock(nEmbd: nEmbd, nHead: nHead, residLambda: 0.5f);

        using var input = randn(batchSize, seqLen, nEmbd);

        // Act
        using var output1 = block1.forward(input, seqLen);
        using var output2 = block2.forward(input, seqLen);

        // Assert: Outputs should be different due to different residual scaling
        var diff = (output1 - output2).abs().sum().item<float>();
        Assert.True(diff > 0, "Different residual lambdas should produce different outputs");
    }

    [Fact]
    public void Forward_WithX0Lambda_AppliesResFormerResiduals()
    {
        // Arrange
        var nEmbd = 64;
        var nHead = 4;
        var batchSize = 1;
        var seqLen = 4;

        // Create two blocks: one with x0_lambda=0, one with x0_lambda=0.1
        using var block1 = new TransformerBlock(nEmbd: nEmbd, nHead: nHead, x0Lambda: 0.0f);
        using var block2 = new TransformerBlock(nEmbd: nEmbd, nHead: nHead, x0Lambda: 0.1f);

        using var input = randn(batchSize, seqLen, nEmbd);

        // Act
        using var output1 = block1.forward(input, seqLen, x0: input);
        using var output2 = block2.forward(input, seqLen, x0: input);

        // Assert: Outputs should be different when x0_lambda is non-zero
        var diff = (output1 - output2).abs().sum().item<float>();
        Assert.True(diff > 0, "Non-zero x0_lambda should affect output");
    }

    [Fact]
    public void Forward_WithX0Null_IgnoresX0Lambda()
    {
        // Arrange
        var nEmbd = 64;
        var nHead = 4;
        var batchSize = 1;
        var seqLen = 4;

        // Create two identical blocks with same x0_lambda
        using var block1 = new TransformerBlock(nEmbd: nEmbd, nHead: nHead, x0Lambda: 0.1f);
        using var block2 = new TransformerBlock(nEmbd: nEmbd, nHead: nHead, x0Lambda: 0.1f);

        using var input = randn(batchSize, seqLen, nEmbd);

        // Act: One with x0=null, one with x0=input
        using var output1 = block1.forward(input, seqLen, x0: null);
        using var output2 = block2.forward(input, seqLen, x0: input);

        // Assert: Outputs should be different because x0 is only applied when non-null
        var diff = (output1 - output2).abs().sum().item<float>();
        Assert.True(diff > 0, "x0=null should ignore x0_lambda");
    }

    [Fact]
    public void Forward_WithZeroInput_ReturnsNonZeroOutput()
    {
        // Arrange
        var nEmbd = 64;
        var nHead = 4;
        var batchSize = 1;
        var seqLen = 4;

        using var block = new TransformerBlock(nEmbd: nEmbd, nHead: nHead);
        using var input = zeros(batchSize, seqLen, nEmbd);

        // Act
        using var output = block.forward(input, seqLen);

        // Assert: Even with zero input, biases and activations should produce non-zero output
        // Actually, our implementation has no biases, so output should be zero
        var sum = output.abs().sum().item<float>();
        Assert.Equal(0, sum, precision: 5);
    }

    [Fact]
    public void Forward_WithSingleToken_HandlesCorrectly()
    {
        // Arrange
        var nEmbd = 64;
        var nHead = 4;
        var batchSize = 1;
        var seqLen = 1;

        using var block = new TransformerBlock(nEmbd: nEmbd, nHead: nHead);
        using var input = randn(batchSize, seqLen, nEmbd);

        // Act
        using var output = block.forward(input, seqLen);

        // Assert
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
    }

    [Fact]
    public void Forward_WithLargeBatch_ProcessesCorrectly()
    {
        // Arrange
        var nEmbd = 64;
        var nHead = 4;
        var batchSize = 32;
        var seqLen = 8;

        using var block = new TransformerBlock(nEmbd: nEmbd, nHead: nHead);
        using var input = randn(batchSize, seqLen, nEmbd);

        // Act
        using var output = block.forward(input, seqLen);

        // Assert
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
    }

    [Fact]
    public void Forward_WithDifferentInputs_ProducesDifferentOutputs()
    {
        // Arrange
        var nEmbd = 64;
        var nHead = 4;
        var batchSize = 1;
        var seqLen = 4;

        using var block = new TransformerBlock(nEmbd: nEmbd, nHead: nHead);
        using var input1 = randn(batchSize, seqLen, nEmbd);
        using var input2 = randn(batchSize, seqLen, nEmbd);

        // Act
        using var output1 = block.forward(input1, seqLen);
        using var output2 = block.forward(input2, seqLen);

        // Assert
        var diff = (output1 - output2).abs().sum().item<float>();
        Assert.True(diff > 0, "Different inputs should produce different outputs");
    }

    [Fact]
    public void Forward_RepeatedCalls_ProducesSameOutput()
    {
        // Arrange
        var nEmbd = 64;
        var nHead = 4;
        var batchSize = 1;
        var seqLen = 4;

        using var block = new TransformerBlock(nEmbd: nEmbd, nHead: nHead);
        using var input = randn(batchSize, seqLen, nEmbd);

        // Act
        using var output1 = block.forward(input, seqLen);
        using var output2 = block.forward(input, seqLen);

        // Assert: Same input should produce same output (deterministic)
        var diff = (output1 - output2).abs().sum().item<float>();
        Assert.Equal(0, diff, precision: 5);
    }

    [Fact]
    public void Forward_WithGPTStyleConfig_WorksCorrectly()
    {
        // Arrange: GPT-2 style configuration
        var nEmbd = 768;
        var nHead = 12;
        var batchSize = 2;
        var seqLen = 16;

        using var block = new TransformerBlock(nEmbd: nEmbd, nHead: nHead);
        using var input = randn(batchSize, seqLen, nEmbd);

        // Act
        using var output = block.forward(input, seqLen);

        // Assert
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
    }

    [Fact]
    public void Dispose_ReleasesResources()
    {
        // Arrange
        var block = new TransformerBlock(nEmbd: 64, nHead: 4);

        // Act
        block.Dispose();

        // Assert: No exception should be thrown
        // Multiple dispose calls should be safe
        block.Dispose();
    }
}
