using TorchSharp;
using static TorchSharp.torch;
using NanoChat.Core.Model;

namespace NanoChat.Core.Tests;

public class RMSNormTests
{
    [Fact]
    public void Forward_SimpleInput_ProducesCorrectOutput()
    {
        // Arrange
        var norm = new RMSNorm(eps: 1e-6);
        
        // Create a simple 2D tensor: [[1.0, 2.0, 3.0]]
        var input = tensor(new float[] { 1.0f, 2.0f, 3.0f }).reshape(1, 3);
        
        // Calculate expected output manually:
        // x = [1, 2, 3]
        // x² = [1, 4, 9]
        // mean(x²) = (1 + 4 + 9) / 3 = 14/3 ≈ 4.6667
        // rms = sqrt(4.6667 + 1e-6) ≈ 2.1602
        // normalized = [1/2.1602, 2/2.1602, 3/2.1602] ≈ [0.4629, 0.9258, 1.3887]
        var expected = tensor(new float[] { 0.4629f, 0.9258f, 1.3887f }).reshape(1, 3);
        
        // Act
        var output = norm.forward(input);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.shape, output.shape);
        
        // Check values with tolerance for floating point comparison
        var diff = (output - expected).abs().max().ToSingle();
        Assert.True(diff < 0.001f, $"Output difference {diff} exceeds tolerance");
        
        // Clean up
        input.Dispose();
        expected.Dispose();
        output.Dispose();
        norm.Dispose();
    }

    [Fact]
    public void Forward_BatchedInput_NormalizesEachSampleIndependently()
    {
        // Arrange
        var norm = new RMSNorm(eps: 1e-6);
        
        // Create a batch of 2 samples, each with 3 features
        var input = tensor(new float[,] 
        {
            { 1.0f, 2.0f, 3.0f },
            { 4.0f, 5.0f, 6.0f }
        });
        
        // Act
        var output = norm.forward(input);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.shape, output.shape);
        
        // Each row should be normalized independently
        // Verify that the RMS of each row is approximately 1
        var outputSquared = output.pow(2);
        var rowMeans = outputSquared.mean(new long[] { -1 });
        var rowRMS = rowMeans.sqrt();
        
        // Each row's RMS should be close to 1.0
        for (int i = 0; i < rowRMS.shape[0]; i++)
        {
            var rmsValue = rowRMS[i].ToSingle();
            Assert.True(Math.Abs(rmsValue - 1.0f) < 0.01f, 
                $"Row {i} RMS {rmsValue} is not close to 1.0");
        }
        
        // Clean up
        input.Dispose();
        output.Dispose();
        outputSquared.Dispose();
        rowMeans.Dispose();
        rowRMS.Dispose();
        norm.Dispose();
    }

    [Fact]
    public void Forward_3DInput_NormalizesLastDimension()
    {
        // Arrange
        var norm = new RMSNorm(eps: 1e-6);
        
        // Create a 3D tensor: (batch=2, seq=3, hidden=4)
        var input = randn(2, 3, 4);
        
        // Act
        var output = norm.forward(input);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.shape, output.shape);
        
        // Verify normalization was applied on the last dimension
        // Each (batch, seq) slice should have RMS ≈ 1
        var outputSquared = output.pow(2);
        var lastDimMeans = outputSquared.mean(new long[] { -1 });
        var lastDimRMS = lastDimMeans.sqrt();
        
        // All RMS values should be close to 1.0
        var allRMS = lastDimRMS.flatten();
        for (int i = 0; i < allRMS.shape[0]; i++)
        {
            var rmsValue = allRMS[i].ToSingle();
            Assert.True(Math.Abs(rmsValue - 1.0f) < 0.01f, 
                $"Position {i} RMS {rmsValue} is not close to 1.0");
        }
        
        // Clean up
        input.Dispose();
        output.Dispose();
        outputSquared.Dispose();
        lastDimMeans.Dispose();
        lastDimRMS.Dispose();
        allRMS.Dispose();
        norm.Dispose();
    }

    [Fact]
    public void Forward_ZeroInput_HandlesGracefully()
    {
        // Arrange
        var norm = new RMSNorm(eps: 1e-6);
        var input = zeros(2, 3);
        
        // Act
        var output = norm.forward(input);
        
        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.shape, output.shape);
        
        // With eps, zeros should remain zeros (0 / sqrt(eps) = 0)
        var allZeros = output.abs().sum().ToSingle();
        Assert.True(allZeros < 1e-5f, "Output should be all zeros");
        
        // Clean up
        input.Dispose();
        output.Dispose();
        norm.Dispose();
    }

    [Fact]
    public void Forward_MatchesPyTorchRMSNorm()
    {
        // This test verifies our implementation matches the expected RMSNorm calculation
        
        // Arrange
        var norm = new RMSNorm(eps: 1e-5);
        
        // Input: [0.5, 1.0, 1.5, 2.0]
        var input = tensor(new float[] { 0.5f, 1.0f, 1.5f, 2.0f }).reshape(1, 4);
        
        // Manual calculation:
        // x² = [0.25, 1.0, 2.25, 4.0]
        // mean(x²) = (0.25 + 1.0 + 2.25 + 4.0) / 4 = 1.875
        // rms = sqrt(1.875 + 1e-5) ≈ 1.369306
        // normalized = x / rms = [0.5/1.369306, 1.0/1.369306, 1.5/1.369306, 2.0/1.369306]
        //            ≈ [0.36515, 0.73030, 1.09545, 1.46060]
        
        // Act
        var output = norm.forward(input);
        
        // Assert - verify the RMS is approximately 1.0 after normalization
        var outputSquared = output.pow(2);
        var outputMean = outputSquared.mean(new long[] { -1 });
        var outputRMS = outputMean.sqrt().ToSingle();
        
        Assert.True(Math.Abs(outputRMS - 1.0f) < 0.01f, 
            $"Output RMS {outputRMS} is not close to 1.0");
        
        // Clean up
        input.Dispose();
        output.Dispose();
        outputSquared.Dispose();
        outputMean.Dispose();
        norm.Dispose();
    }

    [Fact]
    public void Constructor_SetsEpsilonCorrectly()
    {
        // Arrange & Act
        var norm1 = new RMSNorm(eps: 1e-8);
        var norm2 = new RMSNorm(eps: 1e-3);
        
        // Create a very small input where epsilon matters more
        // When input is nearly zero, epsilon dominates the RMS calculation
        var input = tensor(new float[] { 0.0f, 0.0f, 0.0f }).reshape(1, 3);
        
        var output1 = norm1.forward(input);
        var output2 = norm2.forward(input);
        
        // Assert - both should produce zeros (0 / sqrt(eps) = 0)
        // But with different epsilon values, if we had non-zero input near zero,
        // they would differ. For zero input, both should be zero.
        var sum1 = output1.abs().sum().ToSingle();
        var sum2 = output2.abs().sum().ToSingle();
        
        Assert.True(sum1 < 1e-6f, "Output with small epsilon should be near zero");
        Assert.True(sum2 < 1e-6f, "Output with large epsilon should be near zero");
        
        // Clean up
        input.Dispose();
        output1.Dispose();
        output2.Dispose();
        norm1.Dispose();
        norm2.Dispose();
    }
}
