using System.Text;
using System.Text.RegularExpressions;

namespace NanoChat.Core.Tokenizer;

/// <summary>
/// BPE tokenizer implementation compatible with tiktoken format.
/// Uses byte-level BPE encoding with mergeable ranks.
/// </summary>
public class BpeTokenizer : ITokenizer
{
    private readonly Dictionary<byte[], int> _encoder; // byte sequence -> token ID
    private readonly Dictionary<int, byte[]> _decoder; // token ID -> byte sequence
    private readonly Dictionary<string, int> _specialTokens; // special token string -> token ID
    private readonly Dictionary<int, string> _specialTokensDecoder; // token ID -> special token string
    private readonly Regex _pattern;
    
    public int VocabSize { get; }
    public string BosToken => "<|bos|>";
    public string EosToken => "<|eos|>";

    public BpeTokenizer(
        Dictionary<byte[], int> mergeableRanks,
        Dictionary<string, int> specialTokens,
        string pattern = @"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
    {
        _encoder = mergeableRanks;
        _specialTokens = specialTokens;
        _pattern = new Regex(pattern, RegexOptions.Compiled);

        // Build decoder
        _decoder = new Dictionary<int, byte[]>();
        foreach (var kvp in _encoder)
        {
            _decoder[kvp.Value] = kvp.Key;
        }

        _specialTokensDecoder = new Dictionary<int, string>();
        foreach (var kvp in _specialTokens)
        {
            _specialTokensDecoder[kvp.Value] = kvp.Key;
        }

        VocabSize = Math.Max(
            _encoder.Count > 0 ? _encoder.Values.Max() + 1 : 0,
            _specialTokens.Count > 0 ? _specialTokens.Values.Max() + 1 : 0
        );
    }

    public int[] Encode(string text, HashSet<string>? allowedSpecial = null)
    {
        if (string.IsNullOrEmpty(text))
        {
            return Array.Empty<int>();
        }

        var tokens = new List<int>();
        var currentPos = 0;

        while (currentPos < text.Length)
        {
            // Check for special tokens
            var foundSpecial = false;
            if (allowedSpecial != null)
            {
                foreach (var special in allowedSpecial)
                {
                    if (text.Substring(currentPos).StartsWith(special))
                    {
                        if (_specialTokens.TryGetValue(special, out var tokenId))
                        {
                            tokens.Add(tokenId);
                            currentPos += special.Length;
                            foundSpecial = true;
                            break;
                        }
                    }
                }
            }

            if (foundSpecial)
            {
                continue;
            }

            // Find the next text chunk to tokenize
            var match = _pattern.Match(text, currentPos);
            if (match.Success && match.Index == currentPos)
            {
                var chunk = match.Value;
                var chunkTokens = EncodeChunk(chunk);
                tokens.AddRange(chunkTokens);
                currentPos += chunk.Length;
            }
            else
            {
                // Fallback: encode single character
                var chunk = text[currentPos].ToString();
                var chunkTokens = EncodeChunk(chunk);
                tokens.AddRange(chunkTokens);
                currentPos++;
            }
        }

        return tokens.ToArray();
    }

    public string Decode(int[] tokens)
    {
        var bytes = new List<byte>();

        foreach (var token in tokens)
        {
            // Check if it's a special token
            if (_specialTokensDecoder.TryGetValue(token, out var specialToken))
            {
                // For special tokens, encode them as UTF-8 bytes
                bytes.AddRange(Encoding.UTF8.GetBytes(specialToken));
            }
            else if (_decoder.TryGetValue(token, out var tokenBytes))
            {
                bytes.AddRange(tokenBytes);
            }
            else
            {
                // Unknown token - replace with special character
                bytes.AddRange(Encoding.UTF8.GetBytes("�"));
            }
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    private int[] EncodeChunk(string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        
        // Start with individual bytes as tokens
        var tokens = new List<byte[]>();
        foreach (var b in bytes)
        {
            tokens.Add(new byte[] { b });
        }

        // Apply BPE merges
        while (tokens.Count > 1)
        {
            int bestPairIndex = -1;
            int bestRank = int.MaxValue;

            // Find the pair with the lowest rank (highest priority merge)
            for (int i = 0; i < tokens.Count - 1; i++)
            {
                var pair = tokens[i].Concat(tokens[i + 1]).ToArray();
                if (_encoder.TryGetValue(pair, out var rank))
                {
                    if (rank < bestRank)
                    {
                        bestRank = rank;
                        bestPairIndex = i;
                    }
                }
            }

            // If no more merges possible, break
            if (bestPairIndex == -1)
            {
                break;
            }

            // Perform the merge
            var merged = tokens[bestPairIndex].Concat(tokens[bestPairIndex + 1]).ToArray();
            tokens[bestPairIndex] = merged;
            tokens.RemoveAt(bestPairIndex + 1);
        }

        // Convert byte sequences to token IDs
        var result = new List<int>();
        foreach (var token in tokens)
        {
            if (_encoder.TryGetValue(token, out var tokenId))
            {
                result.Add(tokenId);
            }
            else
            {
                // This shouldn't happen if encoder is complete, but handle gracefully
                // Fallback to individual bytes
                foreach (var b in token)
                {
                    if (_encoder.TryGetValue(new byte[] { b }, out var byteTokenId))
                    {
                        result.Add(byteTokenId);
                    }
                }
            }
        }

        return result.ToArray();
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
