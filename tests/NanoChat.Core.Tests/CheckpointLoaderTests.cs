using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using NanoChat.Core.Model;

namespace NanoChat.Core.Tests;

/// <summary>
/// Tests for CheckpointLoader functionality.
/// </summary>
public class CheckpointLoaderTests
{
    [Fact]
    public void LoadStateDict_ThrowsOnNullPath()
    {
        Assert.Throws<ArgumentNullException>(() =>
            CheckpointLoader.LoadStateDict(null!));
    }

    [Fact]
    public void LoadStateDict_ThrowsOnNonExistentFile()
    {
        Assert.Throws<FileNotFoundException>(() =>
            CheckpointLoader.LoadStateDict("nonexistent.pt"));
    }

    [Fact]
    public void MapParameterName_HandlesBlocksConvention()
    {
        // Test cases for parameter name mapping from PyTorch to TorchSharp
        var testCases = new[]
        {
            ("token_embedding.weight", "_tokenEmbedding.weight"),
            ("wte.weight", "_tokenEmbedding.weight"),
            ("blocks.0.attn.q_proj.weight", "block_0._attn._qProj.weight"),
            ("blocks.5.attn.k_proj.weight", "block_5._attn._kProj.weight"),
            ("blocks.10.attn.v_proj.weight", "block_10._attn._vProj.weight"),
            ("blocks.0.attn.out_proj.weight", "block_0._attn._outProj.weight"),
            ("blocks.0.mlp.fc1.weight", "block_0._mlp._fc1.weight"),
            ("blocks.2.mlp.fc2.weight", "block_2._mlp._fc2.weight"),
            ("lm_head.weight", "_lmHead.weight"),
        };

        foreach (var (input, expected) in testCases)
        {
            var result = CheckpointLoader.MapParameterName(input);
            Assert.Equal(expected, result);
        }
    }

    [Fact(Skip = "Requires actual checkpoint file")]
    public void LoadIntoModel_LoadsCheckpointSuccessfully()
    {
        // This test will be enabled once we have a real checkpoint file
        var config = new GPTConfig
        {
            SequenceLen = 2048,
            VocabSize = 32768,
            NLayer = 12,
            NHead = 6,
            NKvHead = 6,
            NEmbd = 768,
            WindowPattern = "SSSL"
        };

        var model = new GPT(config);
        
        // Assuming we have a checkpoint file named "test_checkpoint.pt"
        var checkpointPath = "test_checkpoint.pt";
        
        if (File.Exists(checkpointPath))
        {
            CheckpointLoader.LoadIntoModel(model, checkpointPath, strict: false);
            
            // Verify model is loaded (basic sanity check)
            Assert.NotNull(model);
        }
    }

    [Fact]
    public void CreateSimpleModule_AndSaveLoad()
    {
        // Create a temporary checkpoint path
        var tempPath = Path.GetTempFileName() + ".dat";
        
        try
        {
            // Create a simple GPT model
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

            var model = new GPT(config);
            
            // Save the model
            model.save(tempPath);
            
            // Create a new model with same config
            var model2 = new GPT(config);
            
            // Load weights into the new model
            CheckpointLoader.LoadIntoModel(model2, tempPath, strict: false);
            
            // Basic verification that load succeeded
            Assert.NotNull(model2);
            
            // Cleanup
            model.Dispose();
            model2.Dispose();
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }
}
