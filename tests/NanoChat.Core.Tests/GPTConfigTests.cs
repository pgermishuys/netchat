using NanoChat.Core.Model;
using Xunit;

namespace NanoChat.Core.Tests;

public class GPTConfigTests
{
    [Fact]
    public void Constructor_CreatesDefaultConfig()
    {
        var config = new GPTConfig();

        Assert.Equal(2048, config.SequenceLen);
        Assert.Equal(32768, config.VocabSize);
        Assert.Equal(12, config.NLayer);
        Assert.Equal(6, config.NHead);
        Assert.Equal(6, config.NKvHead);
        Assert.Equal(768, config.NEmbd);
        Assert.Equal("SSSL", config.WindowPattern);
        Assert.Equal(512, config.WindowSize);
        Assert.Null(config.MlpHiddenDim);
        Assert.Null(config.ResidLambdas);
        Assert.Null(config.X0Lambdas);
        Assert.Null(config.ValueEmbeddingPattern);
        Assert.Equal(1e-6f, config.RmsNormEps);
        Assert.Equal(10000f, config.RopeBase);
        Assert.Equal(15.0f, config.LogitSoftcap);
        Assert.Equal(0.0f, config.Dropout);
    }

    [Fact]
    public void CreateNanoChatDefault_MatchesDefaults()
    {
        var config = GPTConfig.CreateNanoChatDefault();

        Assert.Equal(2048, config.SequenceLen);
        Assert.Equal(32768, config.VocabSize);
        Assert.Equal(12, config.NLayer);
        Assert.Equal(6, config.NHead);
        Assert.Equal(6, config.NKvHead);
        Assert.Equal(768, config.NEmbd);
        Assert.Equal("SSSL", config.WindowPattern);
    }

    [Fact]
    public void HeadDim_CalculatesCorrectly()
    {
        var config = new GPTConfig { NEmbd = 768, NHead = 12 };
        Assert.Equal(64, config.HeadDim);

        var config2 = new GPTConfig { NEmbd = 1024, NHead = 16 };
        Assert.Equal(64, config2.HeadDim);
    }

    [Fact]
    public void GetMlpHiddenDim_DefaultsToFourTimesEmbd()
    {
        var config = new GPTConfig { NEmbd = 768 };
        Assert.Equal(3072, config.GetMlpHiddenDim());
    }

    [Fact]
    public void GetMlpHiddenDim_UsesCustomValue()
    {
        var config = new GPTConfig { NEmbd = 768, MlpHiddenDim = 2048 };
        Assert.Equal(2048, config.GetMlpHiddenDim());
    }

    [Fact]
    public void GetResidLambda_DefaultsToOne()
    {
        var config = new GPTConfig { NLayer = 4 };
        Assert.Equal(1.0f, config.GetResidLambda(0));
        Assert.Equal(1.0f, config.GetResidLambda(3));
    }

    [Fact]
    public void GetResidLambda_UsesCustomValues()
    {
        var config = new GPTConfig
        {
            NLayer = 3,
            ResidLambdas = new[] { 0.5f, 0.75f, 1.0f }
        };

        Assert.Equal(0.5f, config.GetResidLambda(0));
        Assert.Equal(0.75f, config.GetResidLambda(1));
        Assert.Equal(1.0f, config.GetResidLambda(2));
    }

    [Fact]
    public void GetX0Lambda_DefaultsToZero()
    {
        var config = new GPTConfig { NLayer = 4 };
        Assert.Equal(0.0f, config.GetX0Lambda(0));
        Assert.Equal(0.0f, config.GetX0Lambda(3));
    }

    [Fact]
    public void GetX0Lambda_UsesCustomValues()
    {
        var config = new GPTConfig
        {
            NLayer = 3,
            X0Lambdas = new[] { 0.1f, 0.2f, 0.3f }
        };

        Assert.Equal(0.1f, config.GetX0Lambda(0));
        Assert.Equal(0.2f, config.GetX0Lambda(1));
        Assert.Equal(0.3f, config.GetX0Lambda(2));
    }

    [Fact]
    public void GetWindowSize_ReturnsCorrectValues()
    {
        var config = new GPTConfig
        {
            NLayer = 8,
            WindowPattern = "SSSL",
            WindowSize = 512
        };

        // Pattern: S S S L S S S L
        Assert.Equal(512, config.GetWindowSize(0)); // S
        Assert.Equal(512, config.GetWindowSize(1)); // S
        Assert.Equal(512, config.GetWindowSize(2)); // S
        Assert.Null(config.GetWindowSize(3));       // L
        Assert.Equal(512, config.GetWindowSize(4)); // S (pattern repeats)
        Assert.Equal(512, config.GetWindowSize(5)); // S
        Assert.Equal(512, config.GetWindowSize(6)); // S
        Assert.Null(config.GetWindowSize(7));       // L
    }

    [Fact]
    public void GetUseValueEmbedding_DefaultsToFalse()
    {
        var config = new GPTConfig { NLayer = 4 };
        Assert.False(config.GetUseValueEmbedding(0));
        Assert.False(config.GetUseValueEmbedding(3));
    }

    [Fact]
    public void GetUseValueEmbedding_UsesPattern()
    {
        var config = new GPTConfig
        {
            NLayer = 8,
            ValueEmbeddingPattern = "YNYN"
        };

        // Pattern: Y N Y N Y N Y N
        Assert.True(config.GetUseValueEmbedding(0));   // Y
        Assert.False(config.GetUseValueEmbedding(1));  // N
        Assert.True(config.GetUseValueEmbedding(2));   // Y
        Assert.False(config.GetUseValueEmbedding(3));  // N
        Assert.True(config.GetUseValueEmbedding(4));   // Y (pattern repeats)
        Assert.False(config.GetUseValueEmbedding(5));  // N
        Assert.True(config.GetUseValueEmbedding(6));   // Y
        Assert.False(config.GetUseValueEmbedding(7));  // N
    }

    [Fact]
    public void Validate_SucceedsWithValidConfig()
    {
        var config = GPTConfig.CreateNanoChatDefault();
        config.Validate(); // Should not throw
    }

    [Fact]
    public void Validate_ThrowsForNonDivisibleEmbdAndHead()
    {
        var config = new GPTConfig { NEmbd = 768, NHead = 7 };
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Validate_ThrowsForNonDivisibleHeadAndKvHead()
    {
        var config = new GPTConfig { NHead = 6, NKvHead = 4 };
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Validate_ThrowsForInvalidResidLambdasLength()
    {
        var config = new GPTConfig
        {
            NLayer = 4,
            ResidLambdas = new[] { 1.0f, 1.0f } // Wrong length
        };

        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Validate_ThrowsForInvalidX0LambdasLength()
    {
        var config = new GPTConfig
        {
            NLayer = 4,
            X0Lambdas = new[] { 0.0f, 0.0f } // Wrong length
        };

        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Validate_ThrowsForInvalidWindowPattern()
    {
        var config = new GPTConfig { WindowPattern = "SSLX" }; // Invalid character
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Validate_ThrowsForInvalidValueEmbeddingPattern()
    {
        var config = new GPTConfig { ValueEmbeddingPattern = "YYNX" }; // Invalid character
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Validate_ThrowsForNegativeValues()
    {
        Assert.Throws<ArgumentException>(() => new GPTConfig { NEmbd = -1 }.Validate());
        Assert.Throws<ArgumentException>(() => new GPTConfig { NHead = 0 }.Validate());
        Assert.Throws<ArgumentException>(() => new GPTConfig { NKvHead = -1 }.Validate());
        Assert.Throws<ArgumentException>(() => new GPTConfig { NLayer = 0 }.Validate());
        Assert.Throws<ArgumentException>(() => new GPTConfig { VocabSize = -1 }.Validate());
        Assert.Throws<ArgumentException>(() => new GPTConfig { SequenceLen = 0 }.Validate());
        Assert.Throws<ArgumentException>(() => new GPTConfig { WindowSize = -1 }.Validate());
    }

    [Fact]
    public void GetWindowSize_ThrowsForOutOfRangeLayer()
    {
        var config = new GPTConfig { NLayer = 4 };
        Assert.Throws<ArgumentOutOfRangeException>(() => config.GetWindowSize(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => config.GetWindowSize(4));
    }

    [Fact]
    public void GetResidLambda_ThrowsForOutOfRangeLayer()
    {
        var config = new GPTConfig { NLayer = 4 };
        Assert.Throws<ArgumentOutOfRangeException>(() => config.GetResidLambda(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => config.GetResidLambda(4));
    }

    [Fact]
    public void GetX0Lambda_ThrowsForOutOfRangeLayer()
    {
        var config = new GPTConfig { NLayer = 4 };
        Assert.Throws<ArgumentOutOfRangeException>(() => config.GetX0Lambda(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => config.GetX0Lambda(4));
    }

    [Fact]
    public void GetUseValueEmbedding_ThrowsForOutOfRangeLayer()
    {
        var config = new GPTConfig { NLayer = 4 };
        Assert.Throws<ArgumentOutOfRangeException>(() => config.GetUseValueEmbedding(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => config.GetUseValueEmbedding(4));
    }
}
