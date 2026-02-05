namespace NanoChat.Core.Tokenizer;

/// <summary>
/// Factory for creating tokenizer instances with nanochat-specific special tokens.
/// </summary>
public static class TokenizerFactory
{
    /// <summary>
    /// Special tokens used in nanochat conversation format.
    /// </summary>
    public static class SpecialTokens
    {
        public const string Bos = "<|bos|>";
        public const string Eos = "<|eos|>";
        public const string UserStart = "<|user_start|>";
        public const string UserEnd = "<|user_end|>";
        public const string AssistantStart = "<|assistant_start|>";
        public const string AssistantEnd = "<|assistant_end|>";
        public const string SystemStart = "<|system_start|>";
        public const string SystemEnd = "<|system_end|>";

        /// <summary>
        /// Gets all special tokens as a set.
        /// </summary>
        public static HashSet<string> All => new()
        {
            Bos, Eos, UserStart, UserEnd, 
            AssistantStart, AssistantEnd, 
            SystemStart, SystemEnd
        };
    }

    /// <summary>
    /// Creates a default nanochat tokenizer with standard special tokens.
    /// Token IDs are assigned starting from the base vocabulary size.
    /// </summary>
    /// <param name="mergeableRanks">BPE merge table (byte sequences -> rank)</param>
    /// <param name="baseVocabSize">Size of base vocabulary (before special tokens)</param>
    /// <returns>BPE tokenizer instance</returns>
    public static BpeTokenizer CreateNanoChatTokenizer(
        Dictionary<byte[], int> mergeableRanks,
        int baseVocabSize = 32000)
    {
        // Assign token IDs to special tokens starting after base vocab
        var specialTokens = new Dictionary<string, int>
        {
            { SpecialTokens.Bos, baseVocabSize },
            { SpecialTokens.Eos, baseVocabSize + 1 },
            { SpecialTokens.UserStart, baseVocabSize + 2 },
            { SpecialTokens.UserEnd, baseVocabSize + 3 },
            { SpecialTokens.AssistantStart, baseVocabSize + 4 },
            { SpecialTokens.AssistantEnd, baseVocabSize + 5 },
            { SpecialTokens.SystemStart, baseVocabSize + 6 },
            { SpecialTokens.SystemEnd, baseVocabSize + 7 }
        };

        return new BpeTokenizer(mergeableRanks, specialTokens);
    }

    /// <summary>
    /// Creates a tokenizer with custom special tokens.
    /// </summary>
    /// <param name="mergeableRanks">BPE merge table (byte sequences -> rank)</param>
    /// <param name="specialTokens">Custom special tokens mapping</param>
    /// <returns>BPE tokenizer instance</returns>
    public static BpeTokenizer CreateCustomTokenizer(
        Dictionary<byte[], int> mergeableRanks,
        Dictionary<string, int> specialTokens)
    {
        return new BpeTokenizer(mergeableRanks, specialTokens);
    }
}
