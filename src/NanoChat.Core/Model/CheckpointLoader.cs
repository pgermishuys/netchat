using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NanoChat.Core.Model;

/// <summary>
/// Loads pretrained weights from PyTorch checkpoint files (.pt, .pth).
/// </summary>
public class CheckpointLoader
{
    /// <summary>
    /// Loads a PyTorch checkpoint file and returns the state dictionary.
    /// </summary>
    /// <param name="checkpointPath">Path to the .pt or .pth checkpoint file.</param>
    /// <returns>Dictionary mapping parameter names to tensors.</returns>
    public static Dictionary<string, Tensor> LoadStateDict(string checkpointPath)
    {
        if (string.IsNullOrWhiteSpace(checkpointPath))
            throw new ArgumentNullException(nameof(checkpointPath));

        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"Checkpoint file not found: {checkpointPath}");

        try
        {
            // TorchSharp's Module.load() loads directly into a module
            // For a more generic state dict loading, we'll use a temporary module approach
            // or parse the pickle file directly
            
            // Create a dictionary to hold the state dict
            var stateDict = new Dictionary<string, Tensor>();
            
            // TorchSharp uses the pickle format for .pt files
            // We'll load using a temporary GPT model and extract its state
            // This is a workaround until we have direct state dict loading
            
            // For now, return an empty dict - this will be populated when we 
            // implement proper pickle parsing or use Module.load()
            Console.WriteLine($"Loading checkpoint from {checkpointPath}...");
            
            return stateDict;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to load checkpoint from {checkpointPath}: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Loads weights into a GPT model from a checkpoint file.
    /// </summary>
    /// <param name="model">The GPT model to load weights into.</param>
    /// <param name="checkpointPath">Path to the checkpoint file.</param>
    /// <param name="strict">If true, requires exact match of all parameter names.</param>
    /// <param name="useNameMapping">If true, applies Python PyTorch to C# TorchSharp name mapping.</param>
    public static void LoadIntoModel(GPT model, string checkpointPath, bool strict = true, bool useNameMapping = false)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        if (string.IsNullOrWhiteSpace(checkpointPath))
            throw new ArgumentNullException(nameof(checkpointPath));

        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"Checkpoint file not found: {checkpointPath}");

        try
        {
            Console.WriteLine($"Loading checkpoint from {checkpointPath}...");
            
            if (!useNameMapping)
            {
                // Direct load - assumes checkpoint naming matches TorchSharp conventions
                model.load(checkpointPath);
                Console.WriteLine("Successfully loaded checkpoint into model.");
            }
            else
            {
                // Custom loading with name mapping for Python PyTorch checkpoints
                LoadWithNameMapping(model, checkpointPath, strict);
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to load checkpoint into model: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Loads weights with custom name mapping from PyTorch to TorchSharp naming.
    /// </summary>
    private static void LoadWithNameMapping(GPT model, string checkpointPath, bool strict)
    {
        // Get model's current parameter names and values
        var modelParams = model.named_parameters().ToDictionary(
            p => p.name,
            p => p.parameter
        );

        Console.WriteLine($"Model has {modelParams.Count} parameters");
        
        // For now, this is a placeholder for custom mapping logic
        // In practice, we'd need to:
        // 1. Load the checkpoint into a temporary buffer
        // 2. Map each checkpoint parameter name to our model's names
        // 3. Copy the tensors into the model
        
        // Since TorchSharp's load() expects the exact naming convention,
        // we may need to create a custom pickle reader or use a different approach
        
        throw new NotImplementedException(
            "Custom name mapping not yet implemented. " +
            "Checkpoint must use TorchSharp naming conventions, or " +
            "convert the checkpoint using Python first.");
    }

    /// <summary>
    /// Maps parameter names from PyTorch checkpoint to C# TorchSharp naming convention.
    /// </summary>
    /// <param name="pytorchName">Parameter name from PyTorch checkpoint.</param>
    /// <returns>Mapped parameter name for C# TorchSharp model.</returns>
    public static string MapParameterName(string pytorchName)
    {
        // PyTorch (Python) naming conventions:
        // - token_embedding.weight or wte.weight
        // - blocks.0.attn.q_proj.weight
        // - blocks.0.attn.k_proj.weight  
        // - blocks.0.attn.v_proj.weight
        // - blocks.0.attn.out_proj.weight
        // - blocks.0.mlp.fc1.weight
        // - blocks.0.mlp.fc2.weight
        // - blocks.0.norm1.weight (if learnable)
        // - blocks.0.norm2.weight (if learnable)
        // - lm_head.weight
        // - final_norm.weight (if learnable)
        
        // TorchSharp (C#) naming conventions:
        // - _tokenEmbedding.weight
        // - block_0._attn._qProj.weight
        // - block_0._attn._kProj.weight
        // - block_0._attn._vProj.weight
        // - block_0._attn._outProj.weight
        // - block_0._mlp._fc1.weight
        // - block_0._mlp._fc2.weight
        // - _lmHead.weight
        
        var name = pytorchName;
        
        // Map token embedding
        if (name.StartsWith("token_embedding.") || name.StartsWith("wte."))
        {
            name = name.Replace("token_embedding.", "_tokenEmbedding.");
            name = name.Replace("wte.", "_tokenEmbedding.");
        }
        
        // Map lm_head
        if (name.StartsWith("lm_head."))
        {
            name = name.Replace("lm_head.", "_lmHead.");
        }
        
        // Map transformer blocks: blocks.N -> block_N
        if (name.Contains("blocks."))
        {
            // Example: blocks.0.attn.q_proj.weight -> block_0.attn.q_proj.weight
            name = System.Text.RegularExpressions.Regex.Replace(
                name,
                @"blocks\.(\d+)",
                "block_$1"
            );
        }
        
        // Map attention projection layers
        name = name.Replace(".attn.", "._attn.");
        name = name.Replace(".q_proj.", "._qProj.");
        name = name.Replace(".k_proj.", "._kProj.");
        name = name.Replace(".v_proj.", "._vProj.");
        name = name.Replace(".out_proj.", "._outProj.");
        
        // Map MLP layers
        name = name.Replace(".mlp.", "._mlp.");
        name = name.Replace(".fc1.", "._fc1.");
        name = name.Replace(".fc2.", "._fc2.");
        
        // Map normalization layers (if they have learnable parameters)
        name = name.Replace(".norm1.", "._norm1.");
        name = name.Replace(".norm2.", "._norm2.");
        name = name.Replace(".final_norm.", "._finalNorm.");
        
        return name;
    }

    /// <summary>
    /// Checks if two tensor shapes match.
    /// </summary>
    private static bool ShapesMatch(long[] shape1, long[] shape2)
    {
        if (shape1.Length != shape2.Length)
            return false;

        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Converts a shape array to a readable string.
    /// </summary>
    private static string ShapeToString(long[] shape)
    {
        return $"[{string.Join(", ", shape)}]";
    }
}
