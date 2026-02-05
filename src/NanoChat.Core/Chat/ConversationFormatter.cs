using NanoChat.Core.Tokenizer;

namespace NanoChat.Core.Chat;

/// <summary>
/// Formats conversation messages into text with nanochat special tokens.
/// Handles conversation rendering with proper token placement.
/// </summary>
public class ConversationFormatter
{
    /// <summary>
    /// Formats a list of messages into a single text string with special tokens.
    /// </summary>
    /// <param name="messages">List of messages to format</param>
    /// <param name="includeAssistantStart">Whether to include assistant start token at the end (for prompting)</param>
    /// <returns>Formatted text with special tokens</returns>
    public static string FormatConversation(IEnumerable<Message> messages, bool includeAssistantStart = true)
    {
        var result = new System.Text.StringBuilder();
        
        // Start with beginning of sequence token
        result.Append(TokenizerFactory.SpecialTokens.Bos);
        
        foreach (var message in messages)
        {
            // Add role-specific tokens around message content
            var (startToken, endToken) = GetRoleTokens(message.Role);
            result.Append(startToken);
            result.Append(message.Content);
            result.Append(endToken);
        }
        
        // If we're prompting for assistant response, add the assistant start token
        if (includeAssistantStart)
        {
            result.Append(TokenizerFactory.SpecialTokens.AssistantStart);
        }
        
        return result.ToString();
    }

    /// <summary>
    /// Formats a single user message into a prompt.
    /// Convenient method for single-turn interactions.
    /// </summary>
    /// <param name="userMessage">The user's message content</param>
    /// <returns>Formatted prompt with special tokens</returns>
    public static string FormatSingleMessage(string userMessage)
    {
        var messages = new[] { new Message { Role = Message.Roles.User, Content = userMessage } };
        return FormatConversation(messages, includeAssistantStart: true);
    }

    /// <summary>
    /// Formats a conversation with a system prompt followed by messages.
    /// </summary>
    /// <param name="systemPrompt">System instruction or context</param>
    /// <param name="messages">User/assistant messages</param>
    /// <param name="includeAssistantStart">Whether to include assistant start token at the end</param>
    /// <returns>Formatted text with special tokens</returns>
    public static string FormatWithSystemPrompt(
        string systemPrompt, 
        IEnumerable<Message> messages, 
        bool includeAssistantStart = true)
    {
        var allMessages = new List<Message>
        {
            new Message { Role = Message.Roles.System, Content = systemPrompt }
        };
        allMessages.AddRange(messages);
        
        return FormatConversation(allMessages, includeAssistantStart);
    }

    /// <summary>
    /// Gets the start and end tokens for a specific role.
    /// </summary>
    private static (string startToken, string endToken) GetRoleTokens(string role)
    {
        return role.ToLowerInvariant() switch
        {
            Message.Roles.System => (TokenizerFactory.SpecialTokens.SystemStart, 
                                     TokenizerFactory.SpecialTokens.SystemEnd),
            Message.Roles.User => (TokenizerFactory.SpecialTokens.UserStart, 
                                   TokenizerFactory.SpecialTokens.UserEnd),
            Message.Roles.Assistant => (TokenizerFactory.SpecialTokens.AssistantStart, 
                                        TokenizerFactory.SpecialTokens.AssistantEnd),
            _ => throw new ArgumentException($"Unknown role: {role}", nameof(role))
        };
    }

    /// <summary>
    /// Wraps assistant response text with appropriate end token.
    /// Used when completing an assistant's response.
    /// </summary>
    /// <param name="responseText">The assistant's generated response</param>
    /// <returns>Response text with end token</returns>
    public static string WrapAssistantResponse(string responseText)
    {
        return responseText + TokenizerFactory.SpecialTokens.AssistantEnd;
    }
}
