using NanoChat.Core.Model;
using NanoChat.Core.Tokenizer;
using TorchSharp;

namespace NanoChat.Core.Chat;

/// <summary>
/// Manages a chat session with conversation history and model interaction.
/// </summary>
public class ChatSession : IDisposable
{
    private readonly GPT _model;
    private readonly ITokenizer _tokenizer;
    private readonly List<Message> _conversationHistory;
    private readonly string? _systemPrompt;
    private bool _disposed;

    /// <summary>
    /// Gets the conversation history.
    /// </summary>
    public IReadOnlyList<Message> History => _conversationHistory.AsReadOnly();

    /// <summary>
    /// Creates a new chat session.
    /// </summary>
    /// <param name="model">The GPT model to use for generation</param>
    /// <param name="tokenizer">The tokenizer for encoding/decoding</param>
    /// <param name="systemPrompt">Optional system prompt to use for all conversations</param>
    public ChatSession(GPT model, ITokenizer tokenizer, string? systemPrompt = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _conversationHistory = new List<Message>();
        _systemPrompt = systemPrompt;
    }

    /// <summary>
    /// Sends a user message and generates an assistant response.
    /// </summary>
    /// <param name="userMessage">The user's message content</param>
    /// <param name="onTokenGenerated">Optional callback for streaming tokens as they're generated</param>
    /// <param name="maxNewTokens">Maximum number of tokens to generate</param>
    /// <param name="temperature">Sampling temperature (higher = more random)</param>
    /// <param name="topK">Top-k filtering (null = no filtering)</param>
    /// <returns>The assistant's response text</returns>
    public string SendMessage(
        string userMessage,
        Action<string>? onTokenGenerated = null,
        int maxNewTokens = 256,
        float temperature = 0.8f,
        int? topK = 40)
    {
        if (string.IsNullOrWhiteSpace(userMessage))
            throw new ArgumentException("User message cannot be empty", nameof(userMessage));

        // Add user message to history
        _conversationHistory.Add(new Message 
        { 
            Role = Message.Roles.User, 
            Content = userMessage 
        });

        // Format the conversation with system prompt if provided
        string prompt;
        if (_systemPrompt != null)
        {
            prompt = ConversationFormatter.FormatWithSystemPrompt(
                _systemPrompt, 
                _conversationHistory, 
                includeAssistantStart: true);
        }
        else
        {
            prompt = ConversationFormatter.FormatConversation(
                _conversationHistory, 
                includeAssistantStart: true);
        }

        // Encode the prompt
        var promptTokens = _tokenizer.Encode(prompt, TokenizerFactory.SpecialTokens.All);
        
        // Convert to tensor
        using var promptTensor = torch.tensor(
            promptTokens.Select(t => (long)t).ToArray(), 
            dtype: torch.ScalarType.Int64)
            .unsqueeze(0); // Add batch dimension

        // Generate response with streaming
        var responseTokens = new List<int>();
        var assistantEndTokenId = _tokenizer.Encode(
            TokenizerFactory.SpecialTokens.AssistantEnd, 
            TokenizerFactory.SpecialTokens.All)[0];
        var eosTokenId = _tokenizer.Encode(
            TokenizerFactory.SpecialTokens.Eos, 
            TokenizerFactory.SpecialTokens.All)[0];

        var stopTokens = new HashSet<long> { assistantEndTokenId, eosTokenId };

        using var generatedTensor = _model.GenerateStreaming(
            promptTensor,
            maxNewTokens: maxNewTokens,
            onTokenGenerated: (batchIdx, tokenId, position) =>
            {
                // Don't include stop tokens in the response
                if (stopTokens.Contains(tokenId))
                    return false;

                responseTokens.Add((int)tokenId);

                // Decode and emit the token if callback provided
                if (onTokenGenerated != null)
                {
                    var tokenText = _tokenizer.Decode(new[] { (int)tokenId });
                    onTokenGenerated(tokenText);
                }

                return true; // Continue generation
            },
            temperature: temperature,
            topK: topK,
            stopTokens: stopTokens,
            useCache: true);

        // Decode the full response
        var responseText = _tokenizer.Decode(responseTokens.ToArray());

        // Add assistant response to history
        _conversationHistory.Add(new Message 
        { 
            Role = Message.Roles.Assistant, 
            Content = responseText 
        });

        return responseText;
    }

    /// <summary>
    /// Clears the conversation history.
    /// </summary>
    public void ClearHistory()
    {
        _conversationHistory.Clear();
    }

    /// <summary>
    /// Gets the token count for the current conversation.
    /// </summary>
    /// <returns>Number of tokens in the conversation</returns>
    public int GetTokenCount()
    {
        string prompt;
        if (_systemPrompt != null)
        {
            prompt = ConversationFormatter.FormatWithSystemPrompt(
                _systemPrompt, 
                _conversationHistory, 
                includeAssistantStart: true);
        }
        else
        {
            prompt = ConversationFormatter.FormatConversation(
                _conversationHistory, 
                includeAssistantStart: true);
        }

        return _tokenizer.Encode(prompt, TokenizerFactory.SpecialTokens.All).Length;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            // Note: We don't dispose the model or tokenizer as they may be used elsewhere
            // The caller is responsible for disposing those resources
            _disposed = true;
        }
    }
}
