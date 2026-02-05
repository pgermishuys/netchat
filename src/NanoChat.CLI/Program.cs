using NanoChat.Core.Tokenizer;
using NanoChat.Core.Model;
using NanoChat.Core.Chat;
using TorchSharp;
using static TorchSharp.torch;

// Parse command line arguments
var mode = args.Length > 0 ? args[0] : "test";

if (mode == "chat")
{
    return RunInteractiveChat();
}
else
{
    return RunTests();
}

// Interactive chat mode
int RunInteractiveChat()
{
    Console.WriteLine("NanoChat - Interactive Mode");
    Console.WriteLine("============================");
    Console.WriteLine();
    Console.WriteLine("Commands:");
    Console.WriteLine("  /exit    - Exit the chat");
    Console.WriteLine("  /clear   - Clear conversation history");
    Console.WriteLine("  /tokens  - Show current token count");
    Console.WriteLine();

    // Setup tokenizer
    var mergeableRanks = new Dictionary<byte[], int>(new ByteArrayComparer());
    for (int i = 0; i < 256; i++)
    {
        mergeableRanks[new byte[] { (byte)i }] = i;
    }
    var tokenizer = TokenizerFactory.CreateNanoChatTokenizer(mergeableRanks);

    // Create a small GPT model for demonstration
    var config = new GPTConfig
    {
        VocabSize = 500,
        NLayer = 2,
        NHead = 4,
        NKvHead = 4,
        NEmbd = 64,
        SequenceLen = 128
    };

    Console.WriteLine($"Loading model: {config.NLayer} layers, {config.NHead} heads, {config.NEmbd} embedding dim");
    using var model = new GPT(config);
    
    var systemPrompt = "You are a helpful AI assistant.";
    using var chatSession = new ChatSession(model, tokenizer, systemPrompt);
    
    Console.WriteLine("Model loaded! Start chatting...");
    Console.WriteLine();

    // Main chat loop
    while (true)
    {
        // Read user input
        Console.ForegroundColor = ConsoleColor.Green;
        Console.Write("You: ");
        Console.ResetColor();
        
        var userInput = Console.ReadLine();
        
        if (string.IsNullOrWhiteSpace(userInput))
            continue;

        // Handle commands
        if (userInput.StartsWith("/"))
        {
            var command = userInput.ToLowerInvariant();
            
            if (command == "/exit")
            {
                Console.WriteLine("Goodbye!");
                return 0;
            }
            else if (command == "/clear")
            {
                chatSession.ClearHistory();
                Console.WriteLine("Conversation history cleared.");
                continue;
            }
            else if (command == "/tokens")
            {
                var tokenCount = chatSession.GetTokenCount();
                Console.WriteLine($"Current token count: {tokenCount}/{config.SequenceLen}");
                continue;
            }
            else
            {
                Console.WriteLine($"Unknown command: {command}");
                continue;
            }
        }

        // Generate assistant response with streaming
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write("Assistant: ");
        Console.ResetColor();

        try
        {
            var response = chatSession.SendMessage(
                userInput,
                onTokenGenerated: (token) =>
                {
                    // Stream tokens as they're generated
                    Console.Write(token);
                },
                maxNewTokens: 100,
                temperature: 0.8f,
                topK: 40
            );

            Console.WriteLine(); // New line after response
            Console.WriteLine();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"Error: {ex.Message}");
            Console.ResetColor();
            Console.WriteLine();
        }
    }
}

// Test mode
int RunTests()
{
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

    Console.WriteLine("\n--- Streaming Generation Demo ---");

    // Create a small GPT model for demonstration
    var config = new GPTConfig
    {
        VocabSize = 500,
        NLayer = 2,
        NHead = 4,
        NKvHead = 4,
        NEmbd = 64,
        SequenceLen = 128
    };

    Console.WriteLine($"Creating GPT model: {config.NLayer} layers, {config.NHead} heads, {config.NEmbd} embedding dim");
    using var model = new GPT(config);

    // Create a test prompt (random token IDs for demo)
    using var prompt = torch.randint(0, 500, new long[] { 1, 10 }, dtype: ScalarType.Int64);
    Console.WriteLine($"Prompt shape: [{prompt.shape[0]}, {prompt.shape[1]}]");

    // Demonstrate streaming generation
    Console.Write("\nGenerating 20 tokens with streaming (each '.' is a token): ");
    var streamedTokens = new List<long>();

    using var generated = model.GenerateStreaming(
        prompt,
        maxNewTokens: 20,
        onTokenGenerated: (batchIdx, tokenId, position) =>
        {
            streamedTokens.Add(tokenId);
            Console.Write(".");
            
            // Simulate processing delay (e.g., writing to output)
            Thread.Sleep(50);
            
            return true; // Continue generation
        },
        temperature: 0.9f,
        useCache: true
    );

    Console.WriteLine($"\n✓ Generated {streamedTokens.Count} tokens");
    Console.WriteLine($"Output shape: [{generated.shape[0]}, {generated.shape[1]}]");

    // Demonstrate early termination
    Console.Write("\nGenerating with early stop after 5 tokens: ");
    var limitedTokens = new List<long>();

    using var limitedGenerated = model.GenerateStreaming(
        prompt,
        maxNewTokens: 20,
        onTokenGenerated: (batchIdx, tokenId, position) =>
        {
            limitedTokens.Add(tokenId);
            Console.Write(".");
            Thread.Sleep(50);
            
            // Stop after 5 tokens
            return position < 4;
        },
        temperature: 0.9f,
        useCache: true
    );

    Console.WriteLine($"\n✓ Generated {limitedTokens.Count} tokens (stopped early)");

    Console.WriteLine("\nAll tests passed!");
    Console.WriteLine($"Tokenizer vocab size: {tokenizer.VocabSize}");
    Console.WriteLine($"Model parameters: ~{config.NLayer * config.NEmbd * config.NEmbd * 8 / 1_000_000:F2}M (estimated)");
    Console.WriteLine("\nTo start interactive chat, run:");
    Console.WriteLine("  dotnet run --project src/NanoChat.CLI chat");
    return 0;
}

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

