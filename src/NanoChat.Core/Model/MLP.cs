using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace NanoChat.Core.Model;

/// <summary>
/// Multi-Layer Perceptron (MLP) block with ReLU² activation
/// 
/// Implements a two-layer feedforward network with ReLU² activation.
/// ReLU² is computed as: relu(x)² - apply ReLU then square the result.
/// 
/// Architecture:
/// - Linear layer: nEmbd -> hiddenDim
/// - ReLU² activation
/// - Linear layer: hiddenDim -> nEmbd
/// </summary>
public class MLP : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _fc1;
    private readonly Module<Tensor, Tensor> _fc2;

    /// <summary>
    /// Initialize MLP block
    /// </summary>
    /// <param name="nEmbd">Embedding dimension (input and output size)</param>
    /// <param name="hiddenDim">Hidden layer dimension. If null, defaults to 4 * nEmbd</param>
    /// <param name="name">Module name</param>
    public MLP(int nEmbd, int? hiddenDim = null, string? name = null) 
        : base(name ?? "MLP")
    {
        var hidden = hiddenDim ?? (4 * nEmbd);
        
        // First linear layer: nEmbd -> hiddenDim
        _fc1 = Linear(nEmbd, hidden, hasBias: false);
        
        // Second linear layer: hiddenDim -> nEmbd
        _fc2 = Linear(hidden, nEmbd, hasBias: false);
        
        // Register modules
        RegisterComponents();
    }

    /// <summary>
    /// Apply MLP transformation to input tensor
    /// </summary>
    /// <param name="x">Input tensor of shape (..., nEmbd)</param>
    /// <returns>Output tensor of same shape as input</returns>
    public override Tensor forward(Tensor x)
    {
        // First linear layer
        var hidden = _fc1.forward(x);
        
        // Apply ReLU² activation: relu(x)²
        // ReLU zeros out negative values, then we square the result
        hidden = F.relu(hidden);
        hidden = hidden.pow(2);
        
        // Second linear layer
        var output = _fc2.forward(hidden);
        
        return output;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _fc1?.Dispose();
            _fc2?.Dispose();
        }
        base.Dispose(disposing);
    }
}
