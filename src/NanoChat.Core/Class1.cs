using TorchSharp;
using static TorchSharp.torch;

namespace NanoChat.Core;

public class TensorExample
{
    public static bool CanCreateTensor()
    {
        using var t = torch.tensor(new[] { 1.0f, 2.0f, 3.0f });
        return t.shape[0] == 3;
    }

    public static bool CanRunMatmul()
    {
        using var a = torch.randn(2, 3);
        using var b = torch.randn(3, 4);
        using var result = torch.matmul(a, b);
        return result.shape[0] == 2 && result.shape[1] == 4;
    }
}
