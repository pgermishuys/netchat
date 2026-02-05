namespace NanoChat.Core.Tests;

public class TensorExampleTests
{
    [Fact]
    public void CanCreateTensor_ReturnsTrue()
    {
        // Act
        var result = TensorExample.CanCreateTensor();

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void CanRunMatmul_ReturnsTrue()
    {
        // Act
        var result = TensorExample.CanRunMatmul();

        // Assert
        Assert.True(result);
    }
}
