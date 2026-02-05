using TorchSharp;
using static TorchSharp.torch;
using NanoChat.Core.Model;

namespace NanoChat.Core.Tests;

/// <summary>
/// Helper test to inspect model parameter names.
/// This helps us understand the naming convention TorchSharp uses.
/// </summary>
public class ModelInspectionTests
{
    [Fact]
    public void InspectGPTParameterNames()
    {
        var config = new GPTConfig
        {
            SequenceLen = 128,
            VocabSize = 1000,
            NLayer = 2,
            NHead = 2,
            NKvHead = 2,
            NEmbd = 64,
            WindowPattern = "SS"
        };

        using var model = new GPT(config);
        
        // Get all parameter names
        var paramNames = model.named_parameters()
            .Select(p => p.name)
            .OrderBy(n => n)
            .ToList();

        // Print them for inspection
        Console.WriteLine("Model Parameters:");
        foreach (var name in paramNames)
        {
            Console.WriteLine($"  {name}");
        }

        // Verify we have parameters
        Assert.NotEmpty(paramNames);
        
        // Check for expected patterns (TorchSharp prefixes private fields with _)
        Assert.Contains(paramNames, n => n.Contains("_tokenEmbedding"));
        Assert.Contains(paramNames, n => n.Contains("_lmHead"));
        Assert.Contains(paramNames, n => n.Contains("block_"));
    }
}
