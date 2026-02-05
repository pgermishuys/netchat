namespace NanoChat.Core.Tokenizer;

/// <summary>
/// Interface for text tokenization compatible with nanochat.
/// Provides encoding/decoding between text and token IDs.
/// </summary>
public interface ITokenizer
{
    /// <summary>
    /// Encodes text into a sequence of token IDs.
    /// </summary>
    /// <param name="text">Input text to tokenize</param>
    /// <param name="allowedSpecial">Set of special tokens allowed in the text (e.g., "&lt;|bos|&gt;", "&lt;|user_start|&gt;")</param>
    /// <returns>Array of token IDs</returns>
    int[] Encode(string text, HashSet<string>? allowedSpecial = null);

    /// <summary>
    /// Decodes a sequence of token IDs back into text.
    /// </summary>
    /// <param name="tokens">Array of token IDs</param>
    /// <returns>Decoded text</returns>
    string Decode(int[] tokens);

    /// <summary>
    /// Gets the vocabulary size of the tokenizer.
    /// </summary>
    int VocabSize { get; }

    /// <summary>
    /// Gets the special token for beginning of sequence.
    /// </summary>
    string BosToken { get; }

    /// <summary>
    /// Gets the special token for end of sequence.
    /// </summary>
    string EosToken { get; }
}
