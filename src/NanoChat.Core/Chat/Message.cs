namespace NanoChat.Core.Chat;

/// <summary>
/// Represents a message in a conversation.
/// </summary>
public record Message
{
    /// <summary>
    /// The role of the message sender (system, user, or assistant).
    /// </summary>
    public required string Role { get; init; }

    /// <summary>
    /// The content of the message.
    /// </summary>
    public required string Content { get; init; }

    /// <summary>
    /// Standard role names.
    /// </summary>
    public static class Roles
    {
        public const string System = "system";
        public const string User = "user";
        public const string Assistant = "assistant";
    }
}
