using NanoChat.Core.Chat;
using NanoChat.Core.Tokenizer;
using Xunit;

namespace NanoChat.Core.Tests;

public class ConversationFormatterTests
{
    [Fact]
    public void FormatConversation_SingleUserMessage_ProducesCorrectFormat()
    {
        // Arrange
        var messages = new[]
        {
            new Message { Role = Message.Roles.User, Content = "Hello!" }
        };

        // Act
        var result = ConversationFormatter.FormatConversation(messages, includeAssistantStart: true);

        // Assert
        var expected = $"{TokenizerFactory.SpecialTokens.Bos}" +
                      $"{TokenizerFactory.SpecialTokens.UserStart}Hello!{TokenizerFactory.SpecialTokens.UserEnd}" +
                      $"{TokenizerFactory.SpecialTokens.AssistantStart}";
        Assert.Equal(expected, result);
    }

    [Fact]
    public void FormatConversation_MultipleMessages_ProducesCorrectFormat()
    {
        // Arrange
        var messages = new[]
        {
            new Message { Role = Message.Roles.User, Content = "What is 2+2?" },
            new Message { Role = Message.Roles.Assistant, Content = "2+2 equals 4." },
            new Message { Role = Message.Roles.User, Content = "Thanks!" }
        };

        // Act
        var result = ConversationFormatter.FormatConversation(messages, includeAssistantStart: true);

        // Assert
        Assert.StartsWith(TokenizerFactory.SpecialTokens.Bos, result);
        Assert.Contains(TokenizerFactory.SpecialTokens.UserStart + "What is 2+2?" + TokenizerFactory.SpecialTokens.UserEnd, result);
        Assert.Contains(TokenizerFactory.SpecialTokens.AssistantStart + "2+2 equals 4." + TokenizerFactory.SpecialTokens.AssistantEnd, result);
        Assert.Contains(TokenizerFactory.SpecialTokens.UserStart + "Thanks!" + TokenizerFactory.SpecialTokens.UserEnd, result);
        Assert.EndsWith(TokenizerFactory.SpecialTokens.AssistantStart, result);
    }

    [Fact]
    public void FormatConversation_WithoutAssistantStart_DoesNotIncludeAssistantStart()
    {
        // Arrange
        var messages = new[]
        {
            new Message { Role = Message.Roles.User, Content = "Hello!" }
        };

        // Act
        var result = ConversationFormatter.FormatConversation(messages, includeAssistantStart: false);

        // Assert
        Assert.DoesNotContain(TokenizerFactory.SpecialTokens.AssistantStart, result.TrimEnd());
        Assert.EndsWith(TokenizerFactory.SpecialTokens.UserEnd, result);
    }

    [Fact]
    public void FormatSingleMessage_ProducesCorrectFormat()
    {
        // Arrange
        var userMessage = "Tell me a joke";

        // Act
        var result = ConversationFormatter.FormatSingleMessage(userMessage);

        // Assert
        var expected = $"{TokenizerFactory.SpecialTokens.Bos}" +
                      $"{TokenizerFactory.SpecialTokens.UserStart}Tell me a joke{TokenizerFactory.SpecialTokens.UserEnd}" +
                      $"{TokenizerFactory.SpecialTokens.AssistantStart}";
        Assert.Equal(expected, result);
    }

    [Fact]
    public void FormatWithSystemPrompt_ProducesCorrectFormat()
    {
        // Arrange
        var systemPrompt = "You are a helpful assistant.";
        var messages = new[]
        {
            new Message { Role = Message.Roles.User, Content = "Hello!" }
        };

        // Act
        var result = ConversationFormatter.FormatWithSystemPrompt(systemPrompt, messages, includeAssistantStart: true);

        // Assert
        Assert.StartsWith(TokenizerFactory.SpecialTokens.Bos, result);
        Assert.Contains(TokenizerFactory.SpecialTokens.SystemStart + systemPrompt + TokenizerFactory.SpecialTokens.SystemEnd, result);
        Assert.Contains(TokenizerFactory.SpecialTokens.UserStart + "Hello!" + TokenizerFactory.SpecialTokens.UserEnd, result);
        Assert.EndsWith(TokenizerFactory.SpecialTokens.AssistantStart, result);
    }

    [Fact]
    public void FormatConversation_SystemMessage_UsesSystemTokens()
    {
        // Arrange
        var messages = new[]
        {
            new Message { Role = Message.Roles.System, Content = "Be concise." }
        };

        // Act
        var result = ConversationFormatter.FormatConversation(messages, includeAssistantStart: false);

        // Assert
        Assert.Contains(TokenizerFactory.SpecialTokens.SystemStart, result);
        Assert.Contains(TokenizerFactory.SpecialTokens.SystemEnd, result);
    }

    [Fact]
    public void FormatConversation_InvalidRole_ThrowsArgumentException()
    {
        // Arrange
        var messages = new[]
        {
            new Message { Role = "invalid_role", Content = "Test" }
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => 
            ConversationFormatter.FormatConversation(messages));
    }

    [Fact]
    public void WrapAssistantResponse_AddsEndToken()
    {
        // Arrange
        var response = "I can help with that.";

        // Act
        var result = ConversationFormatter.WrapAssistantResponse(response);

        // Assert
        var expected = response + TokenizerFactory.SpecialTokens.AssistantEnd;
        Assert.Equal(expected, result);
    }

    [Fact]
    public void FormatConversation_EmptyContent_HandlesCorrectly()
    {
        // Arrange
        var messages = new[]
        {
            new Message { Role = Message.Roles.User, Content = "" }
        };

        // Act
        var result = ConversationFormatter.FormatConversation(messages);

        // Assert
        Assert.Contains(TokenizerFactory.SpecialTokens.UserStart, result);
        Assert.Contains(TokenizerFactory.SpecialTokens.UserEnd, result);
    }

    [Fact]
    public void FormatConversation_MultiTurnConversation_MaintainsOrder()
    {
        // Arrange
        var messages = new[]
        {
            new Message { Role = Message.Roles.System, Content = "You are helpful." },
            new Message { Role = Message.Roles.User, Content = "First question" },
            new Message { Role = Message.Roles.Assistant, Content = "First answer" },
            new Message { Role = Message.Roles.User, Content = "Second question" }
        };

        // Act
        var result = ConversationFormatter.FormatConversation(messages);

        // Assert
        var systemIndex = result.IndexOf(TokenizerFactory.SpecialTokens.SystemStart);
        var firstUserIndex = result.IndexOf(TokenizerFactory.SpecialTokens.UserStart);
        var firstAssistantIndex = result.IndexOf(TokenizerFactory.SpecialTokens.AssistantStart);
        var secondUserIndex = result.LastIndexOf(TokenizerFactory.SpecialTokens.UserStart);

        Assert.True(systemIndex < firstUserIndex);
        Assert.True(firstUserIndex < firstAssistantIndex);
        Assert.True(firstAssistantIndex < secondUserIndex);
    }

    [Fact]
    public void FormatConversation_RoleNamesCaseInsensitive_WorksCorrectly()
    {
        // Arrange
        var messages = new[]
        {
            new Message { Role = "USER", Content = "Test uppercase" },
            new Message { Role = "User", Content = "Test mixed case" },
            new Message { Role = "user", Content = "Test lowercase" }
        };

        // Act
        var result = ConversationFormatter.FormatConversation(messages, includeAssistantStart: false);

        // Assert
        // Should contain user tokens for all three messages
        var userStartCount = result.Split(TokenizerFactory.SpecialTokens.UserStart).Length - 1;
        Assert.Equal(3, userStartCount);
    }

    [Fact]
    public void FormatConversation_EmptyMessageList_ProducesMinimalFormat()
    {
        // Arrange
        var messages = Array.Empty<Message>();

        // Act
        var result = ConversationFormatter.FormatConversation(messages, includeAssistantStart: true);

        // Assert
        var expected = TokenizerFactory.SpecialTokens.Bos + TokenizerFactory.SpecialTokens.AssistantStart;
        Assert.Equal(expected, result);
    }

    [Fact]
    public void FormatWithSystemPrompt_EmptyMessages_IncludesOnlySystemPrompt()
    {
        // Arrange
        var systemPrompt = "System instruction";
        var messages = Array.Empty<Message>();

        // Act
        var result = ConversationFormatter.FormatWithSystemPrompt(systemPrompt, messages);

        // Assert
        Assert.Contains(TokenizerFactory.SpecialTokens.SystemStart, result);
        Assert.Contains(systemPrompt, result);
        Assert.Contains(TokenizerFactory.SpecialTokens.SystemEnd, result);
        Assert.EndsWith(TokenizerFactory.SpecialTokens.AssistantStart, result);
    }
}
