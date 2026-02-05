using NanoChat.Core;

Console.WriteLine("NanoChat - Testing TorchSharp Integration");
Console.WriteLine("==========================================");

Console.Write("Creating tensor... ");
if (TensorExample.CanCreateTensor())
{
    Console.WriteLine("✓ Success");
}
else
{
    Console.WriteLine("✗ Failed");
    return 1;
}

Console.Write("Running matmul... ");
if (TensorExample.CanRunMatmul())
{
    Console.WriteLine("✓ Success");
}
else
{
    Console.WriteLine("✗ Failed");
    return 1;
}

Console.WriteLine("\nAll TorchSharp tests passed!");
return 0;
