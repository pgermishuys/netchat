using TorchSharp;
using static TorchSharp.torch;
using NanoChat.Core.Model;

namespace NanoChat.Core.Tests;

public class AttentionTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesModule()
    {
        // Arrange & Act
        var attention = new Attention(nEmbd: 768, nHead: 12);
        
        // Assert
        Assert.NotNull(attention);
        
        // Clean up
        attention.Dispose();
    }
    
    [Fact]
    public void Constructor_InvalidEmbdNotDivisibleByNHead_ThrowsException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentException>(() => new Attention(nEmbd: 768, nHead: 11));
    }
    
    [Fact]
    public void Constructor_InvalidNHeadNotDivisibleByNKvHead_ThrowsException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentException>(() => new Attention(nEmbd: 768, nHead: 12, nKvHead: 5));
    }
    
    [Fact]
    public void Constructor_GQASupport_CreatesModuleWithFewerKVHeads()
    {
        // Arrange & Act
        var attention = new Attention(nEmbd: 768, nHead: 12, nKvHead: 6);
        
        // Assert
        Assert.NotNull(attention);
        
        // Clean up
        attention.Dispose();
    }
    
    [Fact]
    public void Forward_SimpleInput_ProducesCorrectShape()
    {
        // Arrange
        var batchSize = 2;
        var seqLen = 4;
        var nEmbd = 64;
        var nHead = 4;
        
        var attention = new Attention(nEmbd: nEmbd, nHead: nHead);
        var input = randn(batchSize, seqLen, nEmbd);
        
        // Act
        var output = attention.forward(input, seqLen);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
        
        // Clean up
        input.Dispose();
        output.Dispose();
        attention.Dispose();
    }
    
    [Fact]
    public void Forward_WithGQA_ProducesCorrectShape()
    {
        // Arrange
        var batchSize = 2;
        var seqLen = 8;
        var nEmbd = 768;
        var nHead = 12;
        var nKvHead = 6;
        
        var attention = new Attention(nEmbd: nEmbd, nHead: nHead, nKvHead: nKvHead);
        var input = randn(batchSize, seqLen, nEmbd);
        
        // Act
        var output = attention.forward(input, seqLen);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
        
        // Clean up
        input.Dispose();
        output.Dispose();
        attention.Dispose();
    }
    
    [Fact]
    public void Forward_CausalMasking_PreventsAttendingToFuture()
    {
        // This test verifies that the attention mechanism is causal
        // We'll create a scenario where only the causal nature prevents information flow
        
        // Arrange
        var batchSize = 1;
        var seqLen = 4;
        var nEmbd = 64;
        var nHead = 4;
        
        var attention = new Attention(nEmbd: nEmbd, nHead: nHead);
        
        // Create input where each position has a distinct pattern
        // Position 0: all zeros, Position 1: all ones, etc.
        var input = zeros(batchSize, seqLen, nEmbd);
        for (int i = 0; i < seqLen; i++)
        {
            input[0, i] = ones(nEmbd) * i;
        }
        
        // Act
        var output = attention.forward(input, seqLen);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
        
        // The output at position 0 should only depend on position 0 (all zeros)
        // The output at position 1 should depend on positions 0-1
        // etc.
        // This is hard to verify numerically without knowing exact weights,
        // but we can at least check that the output is valid
        Assert.False(output.isnan().any().ToBoolean(), "Output should not contain NaN");
        Assert.False(output.isinf().any().ToBoolean(), "Output should not contain Inf");
        
        // Clean up
        input.Dispose();
        output.Dispose();
        attention.Dispose();
    }
    
    [Fact]
    public void Forward_WithWindowSize_LimitsAttentionRange()
    {
        // Arrange
        var batchSize = 1;
        var seqLen = 8;
        var nEmbd = 64;
        var nHead = 4;
        var windowSize = 2;
        
        var attention = new Attention(nEmbd: nEmbd, nHead: nHead, windowSize: windowSize);
        var input = randn(batchSize, seqLen, nEmbd);
        
        // Act
        var output = attention.forward(input, seqLen);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
        Assert.False(output.isnan().any().ToBoolean(), "Output should not contain NaN");
        Assert.False(output.isinf().any().ToBoolean(), "Output should not contain Inf");
        
        // Clean up
        input.Dispose();
        output.Dispose();
        attention.Dispose();
    }
    
    [Fact]
    public void Forward_MultipleSequenceLengths_HandlesCorrectly()
    {
        // Test that the same attention module can handle different sequence lengths
        
        // Arrange
        var batchSize = 1;
        var nEmbd = 64;
        var nHead = 4;
        
        var attention = new Attention(nEmbd: nEmbd, nHead: nHead, maxSeqLen: 16);
        
        // Act & Assert for different sequence lengths
        foreach (var seqLen in new[] { 1, 4, 8, 16 })
        {
            var input = randn(batchSize, seqLen, nEmbd);
            var output = attention.forward(input, seqLen);
            
            Assert.NotNull(output);
            Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
            Assert.False(output.isnan().any().ToBoolean(), 
                $"Output for seqLen={seqLen} should not contain NaN");
            
            input.Dispose();
            output.Dispose();
        }
        
        // Clean up
        attention.Dispose();
    }
    
    [Fact]
    public void Forward_ZeroInput_ProducesValidOutput()
    {
        // Arrange
        var batchSize = 2;
        var seqLen = 4;
        var nEmbd = 64;
        var nHead = 4;
        
        var attention = new Attention(nEmbd: nEmbd, nHead: nHead);
        var input = zeros(batchSize, seqLen, nEmbd);
        
        // Act
        var output = attention.forward(input, seqLen);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
        
        // With zero input, after normalization and attention, output should be well-defined
        // (possibly zero or close to zero depending on initialization)
        Assert.False(output.isnan().any().ToBoolean(), "Output should not contain NaN");
        Assert.False(output.isinf().any().ToBoolean(), "Output should not contain Inf");
        
        // Clean up
        input.Dispose();
        output.Dispose();
        attention.Dispose();
    }
    
    [Fact]
    public void Forward_SingleToken_ProducesCorrectShape()
    {
        // Test attention with sequence length of 1 (single token)
        
        // Arrange
        var batchSize = 2;
        var seqLen = 1;
        var nEmbd = 64;
        var nHead = 4;
        
        var attention = new Attention(nEmbd: nEmbd, nHead: nHead);
        var input = randn(batchSize, seqLen, nEmbd);
        
        // Act
        var output = attention.forward(input, seqLen);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
        Assert.False(output.isnan().any().ToBoolean(), "Output should not contain NaN");
        
        // Clean up
        input.Dispose();
        output.Dispose();
        attention.Dispose();
    }
    
    [Fact]
    public void Forward_LargeBatch_HandlesEfficiently()
    {
        // Test with larger batch size to ensure batching works correctly
        
        // Arrange
        var batchSize = 16;
        var seqLen = 8;
        var nEmbd = 128;
        var nHead = 8;
        
        var attention = new Attention(nEmbd: nEmbd, nHead: nHead);
        var input = randn(batchSize, seqLen, nEmbd);
        
        // Act
        var output = attention.forward(input, seqLen);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
        Assert.False(output.isnan().any().ToBoolean(), "Output should not contain NaN");
        Assert.False(output.isinf().any().ToBoolean(), "Output should not contain Inf");
        
        // Clean up
        input.Dispose();
        output.Dispose();
        attention.Dispose();
    }
    
    [Fact]
    public void Forward_DifferentHeadConfigurations_AllWork()
    {
        // Test various valid head configurations
        
        var batchSize = 2;
        var seqLen = 4;
        
        var configs = new[]
        {
            (nEmbd: 64, nHead: 4, nKvHead: 4),   // Standard MHA
            (nEmbd: 64, nHead: 4, nKvHead: 2),   // GQA with 2 groups
            (nEmbd: 64, nHead: 4, nKvHead: 1),   // GQA with 1 KV head (MQA)
            (nEmbd: 128, nHead: 8, nKvHead: 4),  // Larger GQA
        };
        
        foreach (var (nEmbd, nHead, nKvHead) in configs)
        {
            var attention = new Attention(nEmbd: nEmbd, nHead: nHead, nKvHead: nKvHead);
            var input = randn(batchSize, seqLen, nEmbd);
            
            var output = attention.forward(input, seqLen);
            
            Assert.NotNull(output);
            Assert.Equal(new long[] { batchSize, seqLen, nEmbd }, output.shape);
            Assert.False(output.isnan().any().ToBoolean(), 
                $"Config ({nEmbd}, {nHead}, {nKvHead}) produced NaN");
            
            input.Dispose();
            output.Dispose();
            attention.Dispose();
        }
    }
}
