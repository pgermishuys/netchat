using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NanoChat.Core.Model;

/// <summary>
/// Root Mean Square Layer Normalization
/// 
/// RMSNorm normalizes the input using the root mean square statistic.
/// This implementation has no learnable parameters, matching nanochat's F.rms_norm() usage.
/// 
/// Formula: RMSNorm(x) = x / sqrt(mean(x²) + eps)
/// </summary>
public class RMSNorm : Module<Tensor, Tensor>
{
    private readonly double _eps;

    /// <summary>
    /// Initialize RMSNorm layer
    /// </summary>
    /// <param name="eps">Small constant for numerical stability (default: 1e-6)</param>
    /// <param name="name">Module name</param>
    public RMSNorm(double eps = 1e-6, string? name = null) : base(name ?? "RMSNorm")
    {
        _eps = eps;
    }

    /// <summary>
    /// Apply RMS normalization to input tensor
    /// </summary>
    /// <param name="x">Input tensor of shape (..., hidden_size)</param>
    /// <returns>Normalized tensor of same shape as input</returns>
    public override Tensor forward(Tensor x)
    {
        // Compute RMS: sqrt(mean(x²) + eps)
        // We normalize over the last dimension
        var variance = x.pow(2).mean(new long[] { -1 }, keepdim: true);
        var rms = (variance + _eps).sqrt();
        
        // Normalize: x / rms
        return x / rms;
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
    }
}
