using System.Text;
using System.Text.Json;

namespace NanoChat.Core.Tokenizer;

/// <summary>
/// Loads tokenizer data from disk in various formats.
/// </summary>
public static class TokenizerLoader
{
    /// <summary>
    /// Loads a tokenizer from a tiktoken-style format.
    /// The file should contain base64-encoded mergeable ranks (one per line: "base64bytes rank").
    /// </summary>
    /// <param name="mergeableRanksPath">Path to the mergeable ranks file</param>
    /// <param name="baseVocabSize">Base vocabulary size (before special tokens)</param>
    /// <returns>Loaded tokenizer instance</returns>
    public static ITokenizer LoadFromTiktoken(string mergeableRanksPath, int baseVocabSize = 32000)
    {
        var mergeableRanks = LoadMergeableRanks(mergeableRanksPath);
        return TokenizerFactory.CreateNanoChatTokenizer(mergeableRanks, baseVocabSize);
    }

    /// <summary>
    /// Loads mergeable ranks from a tiktoken-style file.
    /// Format: Each line contains "base64bytes rank"
    /// </summary>
    private static Dictionary<byte[], int> LoadMergeableRanks(string path)
    {
        var ranks = new Dictionary<byte[], int>(new ByteArrayComparer());

        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var parts = line.Split(' ');
            if (parts.Length != 2)
                continue;

            try
            {
                var bytes = Convert.FromBase64String(parts[0]);
                var rank = int.Parse(parts[1]);
                ranks[bytes] = rank;
            }
            catch
            {
                // Skip invalid lines
                continue;
            }
        }

        return ranks;
    }

    /// <summary>
    /// Loads a tokenizer from a JSON format.
    /// Expected format:
    /// {
    ///   "mergeable_ranks": { "base64bytes": rank, ... },
    ///   "special_tokens": { "token_string": token_id, ... }
    /// }
    /// </summary>
    public static ITokenizer LoadFromJson(string jsonPath)
    {
        var json = File.ReadAllText(jsonPath);
        var data = JsonSerializer.Deserialize<TokenizerData>(json);

        if (data == null)
            throw new InvalidDataException("Failed to deserialize tokenizer JSON");

        // Convert base64 encoded ranks to byte arrays
        var mergeableRanks = new Dictionary<byte[], int>(new ByteArrayComparer());
        if (data.MergeableRanks != null)
        {
            foreach (var kvp in data.MergeableRanks)
            {
                var bytes = Convert.FromBase64String(kvp.Key);
                mergeableRanks[bytes] = kvp.Value;
            }
        }

        var specialTokens = data.SpecialTokens ?? new Dictionary<string, int>();
        
        return new BpeTokenizer(mergeableRanks, specialTokens);
    }

    /// <summary>
    /// Saves a tokenizer to JSON format for easier loading.
    /// </summary>
    public static void SaveToJson(BpeTokenizer tokenizer, string jsonPath)
    {
        // Note: This would require exposing internal data from BpeTokenizer
        // For now, this is a placeholder for future implementation
        throw new NotImplementedException("Tokenizer serialization not yet implemented");
    }

    // Helper class for JSON deserialization
    private class TokenizerData
    {
        public Dictionary<string, int>? MergeableRanks { get; set; }
        public Dictionary<string, int>? SpecialTokens { get; set; }
    }

    // Custom comparer for byte arrays as dictionary keys
    private class ByteArrayComparer : IEqualityComparer<byte[]>
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
}
