using NanoChat.Core.Tokenizer;
using TorchSharp;
using static TorchSharp.torch;

Console.WriteLine("NanoChat - .NET Implementation");
Console.WriteLine("================================");

// Test TorchSharp integration
Console.Write("Testing TorchSharp tensor creation... ");
using var t = torch.tensor(new[] { 1.0f, 2.0f, 3.0f });
if (t.shape[0] == 3)
{
    Console.WriteLine("✓ Success");
}
else
{
    Console.WriteLine("✗ Failed");
    return 1;
}

Console.Write("Testing TorchSharp matmul... ");
using var a = torch.randn(2, 3);
using var b = torch.randn(3, 4);
using var result = torch.matmul(a, b);
if (result.shape[0] == 2 && result.shape[1] == 4)
{
    Console.WriteLine("✓ Success");
}
else
{
    Console.WriteLine("✗ Failed");
    return 1;
}

// Test Tokenizer
Console.Write("Testing tokenizer... ");
var mergeableRanks = new Dictionary<byte[], int>(new ByteArrayComparer());

// Add individual bytes as base tokens
for (int i = 0; i < 256; i++)
{
    mergeableRanks[new byte[] { (byte)i }] = i;
}

var tokenizer = TokenizerFactory.CreateNanoChatTokenizer(mergeableRanks);
var testText = "Hello, world!";
var tokens = tokenizer.Encode(testText);
var decoded = tokenizer.Decode(tokens);

if (decoded == testText)
{
    Console.WriteLine("✓ Success");
}
else
{
    Console.WriteLine("✗ Failed");
    return 1;
}

Console.WriteLine("\nAll tests passed!");
Console.WriteLine($"Tokenizer vocab size: {tokenizer.VocabSize}");
return 0;

// Helper class for byte array comparison
class ByteArrayComparer : IEqualityComparer<byte[]>
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

