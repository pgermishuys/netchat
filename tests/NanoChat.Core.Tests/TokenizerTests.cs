using NanoChat.Core.Tokenizer;
using System.Text;

namespace NanoChat.Core.Tests;

public class TokenizerTests
{
    private static BpeTokenizer CreateTestTokenizer()
    {
        // Create a simple test tokenizer with basic BPE rules
        // This demonstrates the tokenizer's functionality with a minimal vocabulary
        var mergeableRanks = new Dictionary<byte[], int>(new ByteArrayComparer());
        
        // Add individual bytes (0-255) as base tokens
        for (int i = 0; i < 256; i++)
        {
            mergeableRanks[new byte[] { (byte)i }] = i;
        }

        // Add some common multi-byte sequences with higher ranks (lower priority)
        // For example, common character combinations
        var commonPairs = new Dictionary<string, int>
        {
            { "th", 256 },
            { "he", 257 },
            { "in", 258 },
            { "er", 259 },
            { "an", 260 },
            { " t", 261 },
            { " a", 262 },
            { "ing", 263 }
        };

        foreach (var pair in commonPairs)
        {
            var bytes = Encoding.UTF8.GetBytes(pair.Key);
            mergeableRanks[bytes] = pair.Value;
        }

        return TokenizerFactory.CreateNanoChatTokenizer(mergeableRanks, baseVocabSize: 32000);
    }

    [Fact]
    public void TestBasicEncoding()
    {
        var tokenizer = CreateTestTokenizer();
        var text = "hello";
        var tokens = tokenizer.Encode(text);

        Assert.NotEmpty(tokens);
        Assert.All(tokens, token => Assert.True(token >= 0 && token < tokenizer.VocabSize));
    }

    [Fact]
    public void TestBasicDecoding()
    {
        var tokenizer = CreateTestTokenizer();
        var text = "hello world";
        
        var tokens = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(tokens);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void TestEncodeDecodeRoundtrip()
    {
        var tokenizer = CreateTestTokenizer();
        var testStrings = new[]
        {
            "hello",
            "hello world",
            "The quick brown fox",
            "Testing 123",
            "Special chars: !@#$%",
            ""
        };

        foreach (var text in testStrings)
        {
            var tokens = tokenizer.Encode(text);
            var decoded = tokenizer.Decode(tokens);
            Assert.Equal(text, decoded);
        }
    }

    [Fact]
    public void TestSpecialTokens()
    {
        var tokenizer = CreateTestTokenizer();
        
        // Test encoding with special tokens
        var text = $"{TokenizerFactory.SpecialTokens.Bos}Hello{TokenizerFactory.SpecialTokens.Eos}";
        var allowedSpecial = TokenizerFactory.SpecialTokens.All;
        
        var tokens = tokenizer.Encode(text, allowedSpecial);
        
        Assert.NotEmpty(tokens);
        // First token should be BOS
        Assert.True(tokens[0] >= 32000); // Special tokens start at 32000
        // Last token should be EOS
        Assert.True(tokens[^1] >= 32000);
    }

    [Fact]
    public void TestSpecialTokenDecoding()
    {
        var tokenizer = CreateTestTokenizer();
        
        var text = $"{TokenizerFactory.SpecialTokens.Bos}Hello{TokenizerFactory.SpecialTokens.Eos}";
        var allowedSpecial = TokenizerFactory.SpecialTokens.All;
        
        var tokens = tokenizer.Encode(text, allowedSpecial);
        var decoded = tokenizer.Decode(tokens);
        
        Assert.Equal(text, decoded);
    }

    [Fact]
    public void TestEmptyString()
    {
        var tokenizer = CreateTestTokenizer();
        var tokens = tokenizer.Encode("");
        
        Assert.Empty(tokens);
    }

    [Fact]
    public void TestVocabSize()
    {
        var tokenizer = CreateTestTokenizer();
        
        // Should have base vocab (32000) + 8 special tokens
        Assert.True(tokenizer.VocabSize >= 32008);
    }

    [Fact]
    public void TestAllSpecialTokens()
    {
        var tokenizer = CreateTestTokenizer();
        var specialTokens = new[]
        {
            TokenizerFactory.SpecialTokens.Bos,
            TokenizerFactory.SpecialTokens.Eos,
            TokenizerFactory.SpecialTokens.UserStart,
            TokenizerFactory.SpecialTokens.UserEnd,
            TokenizerFactory.SpecialTokens.AssistantStart,
            TokenizerFactory.SpecialTokens.AssistantEnd,
            TokenizerFactory.SpecialTokens.SystemStart,
            TokenizerFactory.SpecialTokens.SystemEnd
        };

        var allowedSpecial = TokenizerFactory.SpecialTokens.All;

        foreach (var special in specialTokens)
        {
            var tokens = tokenizer.Encode(special, allowedSpecial);
            Assert.Single(tokens);
            
            var decoded = tokenizer.Decode(tokens);
            Assert.Equal(special, decoded);
        }
    }

    [Fact]
    public void TestConversationFormat()
    {
        var tokenizer = CreateTestTokenizer();
        
        // Simulate a conversation format
        var conversation = $"{TokenizerFactory.SpecialTokens.Bos}" +
                          $"{TokenizerFactory.SpecialTokens.UserStart}Hello, how are you?{TokenizerFactory.SpecialTokens.UserEnd}" +
                          $"{TokenizerFactory.SpecialTokens.AssistantStart}I'm doing well, thank you!{TokenizerFactory.SpecialTokens.AssistantEnd}" +
                          $"{TokenizerFactory.SpecialTokens.Eos}";
        
        var allowedSpecial = TokenizerFactory.SpecialTokens.All;
        var tokens = tokenizer.Encode(conversation, allowedSpecial);
        var decoded = tokenizer.Decode(tokens);
        
        Assert.Equal(conversation, decoded);
    }

    [Fact]
    public void TestUnicodeHandling()
    {
        var tokenizer = CreateTestTokenizer();
        var unicodeTexts = new[]
        {
            "Hello 世界",
            "emoji: 😀🎉",
            "Привет мир",
            "مرحبا"
        };

        foreach (var text in unicodeTexts)
        {
            var tokens = tokenizer.Encode(text);
            var decoded = tokenizer.Decode(tokens);
            Assert.Equal(text, decoded);
        }
    }

    // Helper class for byte array comparison
    private class ByteArrayComparer : IEqualityComparer<byte[]>
    {
        public bool Equals(byte[]? x, byte[]? y)
        {
            if (x == null && y == null) return true;
            if (x == null || y == null) return false;
            return x.SequenceEqual(y);
        }

        public int GetHashCode(byte[] obj)
        {
            if (obj == null) return 0;
            
            int hash = 17;
            foreach (var b in obj)
            {
                hash = hash * 31 + b;
            }
            return hash;
        }
    }
}
