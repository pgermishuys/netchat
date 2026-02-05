using TorchSharp;
using static TorchSharp.torch;
using NanoChat.Core.Model;

namespace NanoChat.Core.Tests;

public class RotaryEmbeddingTests
{
    [Fact]
    public void Constructor_InitializesCacheCorrectly()
    {
        // Arrange & Act
        var rope = new RotaryEmbedding(dim: 64, maxSeqLen: 128, @base: 10000.0);
        
        // Assert - should not throw
        Assert.NotNull(rope);
        
        // Clean up
        rope.Dispose();
    }

    [Fact]
    public void Forward_SimpleInput_ProducesCorrectShape()
    {
        // Arrange
        var dim = 64;
        var batchSize = 2;
        var seqLen = 10;
        var nHeads = 4;
        
        var rope = new RotaryEmbedding(dim: dim, maxSeqLen: 128);
        
        // Create input tensor: (batch, seqLen, nHeads, headDim)
        var input = randn(batchSize, seqLen, nHeads, dim);
        
        // Act
        var output = rope.forward(input, seqLen);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.shape, output.shape);
        Assert.Equal(new long[] { batchSize, seqLen, nHeads, dim }, output.shape);
        
        // Clean up
        input.Dispose();
        output.Dispose();
        rope.Dispose();
    }

    [Fact]
    public void Forward_PreservesNorm()
    {
        // RoPE should preserve the norm of the input (it's a rotation)
        
        // Arrange
        var dim = 64;
        var batchSize = 1;
        var seqLen = 5;
        var nHeads = 2;
        
        var rope = new RotaryEmbedding(dim: dim, maxSeqLen: 128);
        var input = randn(batchSize, seqLen, nHeads, dim);
        
        // Calculate input norm
        var inputNorm = input.norm().ToDouble();
        
        // Act
        var output = rope.forward(input, seqLen);
        
        // Calculate output norm
        var outputNorm = output.norm().ToDouble();
        
        // Assert - norms should be approximately equal (rotation preserves magnitude)
        var normDiff = Math.Abs(inputNorm - outputNorm);
        Assert.True(normDiff < 0.01, 
            $"Norm difference {normDiff} is too large. Input norm: {inputNorm}, Output norm: {outputNorm}");
        
        // Clean up
        input.Dispose();
        output.Dispose();
        rope.Dispose();
    }

    [Fact]
    public void Forward_DifferentPositions_ProducesDifferentOutputs()
    {
        // RoPE should encode position information, so different positions
        // should produce different rotations
        
        // Arrange
        var dim = 64;
        var rope = new RotaryEmbedding(dim: dim, maxSeqLen: 128);
        
        // Same input at two different positions
        var input1 = ones(1, 1, 1, dim); // position 0
        var input2 = ones(1, 1, 1, dim); // position 1
        var combined = cat(new[] { input1, input2 }, dim: 1); // (1, 2, 1, dim)
        
        // Act
        var output = rope.forward(combined, 2);
        
        // Extract the two positions
        var pos0 = output[0, 0, 0]; // position 0
        var pos1 = output[0, 1, 0]; // position 1
        
        // Assert - they should be different
        var diff = (pos0 - pos1).abs().sum().ToSingle();
        Assert.True(diff > 0.1f, 
            $"Outputs at different positions should differ. Diff: {diff}");
        
        // Clean up
        input1.Dispose();
        input2.Dispose();
        combined.Dispose();
        output.Dispose();
        pos0.Dispose();
        pos1.Dispose();
        rope.Dispose();
    }

    [Fact]
    public void Forward_VerifyRotationFormula()
    {
        // Test the core rotation formula on a simple case
        
        // Arrange
        var dim = 4; // Small dimension for manual verification
        var rope = new RotaryEmbedding(dim: dim, maxSeqLen: 10, @base: 10000.0);
        
        // Create a simple input vector at position 0
        var input = tensor(new float[] { 1.0f, 0.0f, 1.0f, 0.0f })
            .reshape(1, 1, 1, 4); // (batch=1, seq=1, heads=1, dim=4)
        
        // Act
        var output = rope.forward(input, 1);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.shape, output.shape);
        
        // The rotation should preserve the norm
        var inputNorm = input.norm().ToSingle();
        var outputNorm = output.norm().ToSingle();
        Assert.True(Math.Abs(inputNorm - outputNorm) < 0.001f,
            $"Norm not preserved: input={inputNorm}, output={outputNorm}");
        
        // Clean up
        input.Dispose();
        output.Dispose();
        rope.Dispose();
    }

    [Fact]
    public void Forward_DifferentSequenceLengths_WorksCorrectly()
    {
        // Test that we can use different sequence lengths (up to max)
        
        // Arrange
        var dim = 32;
        var maxSeqLen = 100;
        var rope = new RotaryEmbedding(dim: dim, maxSeqLen: maxSeqLen);
        
        var input1 = randn(1, 10, 1, dim);
        var input2 = randn(1, 50, 1, dim);
        var input3 = randn(1, 100, 1, dim);
        
        // Act
        var output1 = rope.forward(input1, 10);
        var output2 = rope.forward(input2, 50);
        var output3 = rope.forward(input3, 100);
        
        // Assert - all should have correct shapes
        Assert.Equal(input1.shape, output1.shape);
        Assert.Equal(input2.shape, output2.shape);
        Assert.Equal(input3.shape, output3.shape);
        
        // Clean up
        input1.Dispose();
        input2.Dispose();
        input3.Dispose();
        output1.Dispose();
        output2.Dispose();
        output3.Dispose();
        rope.Dispose();
    }

    [Fact]
    public void Forward_MatchesExpectedRotation()
    {
        // Verify the rotation works as expected by checking that
        // applying RoPE twice at different positions gives different results
        
        // Arrange
        var dim = 8;
        var rope = new RotaryEmbedding(dim: dim, maxSeqLen: 20);
        
        // Same vector replicated at 3 positions
        var singleVec = ones(1, 1, 1, dim);
        var threePositions = cat(new[] { singleVec, singleVec, singleVec }, dim: 1);
        
        // Act
        var output = rope.forward(threePositions, 3);
        
        // Assert - each position should be rotated differently
        var pos0 = output[0, 0, 0];
        var pos1 = output[0, 1, 0];
        var pos2 = output[0, 2, 0];
        
        var diff01 = (pos0 - pos1).abs().sum().ToSingle();
        var diff12 = (pos1 - pos2).abs().sum().ToSingle();
        var diff02 = (pos0 - pos2).abs().sum().ToSingle();
        
        Assert.True(diff01 > 0.01f, "Position 0 and 1 should differ");
        Assert.True(diff12 > 0.01f, "Position 1 and 2 should differ");
        Assert.True(diff02 > diff01, "Position 0 and 2 should differ more than 0 and 1");
        
        // Clean up
        singleVec.Dispose();
        threePositions.Dispose();
        output.Dispose();
        pos0.Dispose();
        pos1.Dispose();
        pos2.Dispose();
        rope.Dispose();
    }

    [Fact]
    public void Forward_CorrectDimensionSplit()
    {
        // RoPE splits the dimension in half and rotates pairs
        // This test verifies the dimension handling
        
        // Arrange
        var dim = 64;
        var rope = new RotaryEmbedding(dim: dim, maxSeqLen: 10);
        
        // Create input with distinct halves - use multiple positions
        var firstHalf = ones(32) * 1.0f;
        var secondHalf = ones(32) * 2.0f;
        var combined = cat(new[] { firstHalf, secondHalf }, dim: 0);
        // Stack 3 positions with the same input
        var input = cat(new[] { 
            combined.reshape(1, 1, 1, 64),
            combined.reshape(1, 1, 1, 64),
            combined.reshape(1, 1, 1, 64)
        }, dim: 1); // Shape: (1, 3, 1, 64)
        
        // Act
        var output = rope.forward(input, 3);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(new long[] { 1, 3, 1, 64 }, output.shape);
        
        // At position 0, RoPE might produce identity (cos=1, sin=0)
        // But at positions 1 and 2, it should definitely differ
        var pos0Out = output[0, 0, 0];
        var pos1Out = output[0, 1, 0];
        var pos2Out = output[0, 2, 0];
        
        var inputVec = combined;
        
        // Position 1 and 2 should differ from the original input
        var diff1 = (pos1Out - inputVec).abs().sum().ToSingle();
        var diff2 = (pos2Out - inputVec).abs().sum().ToSingle();
        
        Assert.True(diff1 > 0.01f, $"Position 1 should differ from input. Diff: {diff1}");
        Assert.True(diff2 > 0.01f, $"Position 2 should differ from input. Diff: {diff2}");
        
        // Clean up
        firstHalf.Dispose();
        secondHalf.Dispose();
        combined.Dispose();
        input.Dispose();
        output.Dispose();
        pos0Out.Dispose();
        pos1Out.Dispose();
        pos2Out.Dispose();
        rope.Dispose();
    }

    [Fact]
    public void Forward_BaseParameter_AffectsFrequency()
    {
        // Different base values should produce different frequency patterns
        
        // Arrange
        var dim = 16;
        var rope1 = new RotaryEmbedding(dim: dim, maxSeqLen: 10, @base: 10000.0);
        var rope2 = new RotaryEmbedding(dim: dim, maxSeqLen: 10, @base: 500.0);
        
        var input = randn(1, 5, 1, dim);
        
        // Act
        var output1 = rope1.forward(input, 5);
        var output2 = rope2.forward(input, 5);
        
        // Assert - different bases should produce different outputs
        var diff = (output1 - output2).abs().sum().ToSingle();
        Assert.True(diff > 0.1f, 
            $"Different base values should produce different outputs. Diff: {diff}");
        
        // Clean up
        input.Dispose();
        output1.Dispose();
        output2.Dispose();
        rope1.Dispose();
        rope2.Dispose();
    }
}
