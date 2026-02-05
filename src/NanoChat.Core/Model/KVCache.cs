using TorchSharp;
using static TorchSharp.torch;

namespace NanoChat.Core.Model;

/// <summary>
/// Key-Value cache for efficient autoregressive inference.
/// 
/// During generation, we can cache the key and value tensors from previous tokens
/// to avoid recomputing them. This dramatically speeds up inference.
/// 
/// For each layer and head, we store:
/// - Keys: (batch, nHead, cachedLen, headDim)
/// - Values: (batch, nHead, cachedLen, headDim)
/// 
/// When generating a new token, we:
/// 1. Append new K/V to the cache
/// 2. Use the full cached K/V for attention
/// 3. Only compute Q for the new token
/// </summary>
public class KVCache : IDisposable
{
    private readonly int _nLayers;
    private readonly int _nHeads;
    private readonly int _headDim;
    private readonly int _batchSize;
    private readonly int _maxSeqLen;
    
    // Cache storage: [layer][0=key, 1=value]
    private readonly Tensor?[][] _cache;
    private bool _disposed;

    /// <summary>
    /// Gets the current cache length (number of cached tokens).
    /// </summary>
    public int CacheLength { get; private set; }

    /// <summary>
    /// Creates a new KV cache.
    /// </summary>
    /// <param name="nLayers">Number of transformer layers</param>
    /// <param name="nHeads">Number of attention heads (after GQA expansion)</param>
    /// <param name="headDim">Dimension of each attention head</param>
    /// <param name="batchSize">Batch size</param>
    /// <param name="maxSeqLen">Maximum sequence length</param>
    public KVCache(int nLayers, int nHeads, int headDim, int batchSize, int maxSeqLen)
    {
        _nLayers = nLayers;
        _nHeads = nHeads;
        _headDim = headDim;
        _batchSize = batchSize;
        _maxSeqLen = maxSeqLen;
        
        // Initialize cache storage
        _cache = new Tensor?[nLayers][];
        for (int i = 0; i < nLayers; i++)
        {
            _cache[i] = new Tensor?[2]; // [key, value]
        }
        
        CacheLength = 0;
    }

    /// <summary>
    /// Updates the cache for a specific layer by appending new keys and values.
    /// </summary>
    /// <param name="layerIdx">Layer index (0-based)</param>
    /// <param name="newKeys">New keys to append, shape (batch, nHead, newSeqLen, headDim)</param>
    /// <param name="newValues">New values to append, shape (batch, nHead, newSeqLen, headDim)</param>
    /// <returns>Tuple of (cachedKeys, cachedValues) with full history. These are aliases to the cached tensors, do not dispose them.</returns>
    public (Tensor keys, Tensor values) Update(int layerIdx, Tensor newKeys, Tensor newValues)
    {
        if (layerIdx < 0 || layerIdx >= _nLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIdx));

        if (newKeys.shape[0] != _batchSize || newValues.shape[0] != _batchSize)
            throw new ArgumentException("Batch size mismatch");

        if (newKeys.shape[1] != _nHeads || newValues.shape[1] != _nHeads)
            throw new ArgumentException("Number of heads mismatch");

        if (newKeys.shape[3] != _headDim || newValues.shape[3] != _headDim)
            throw new ArgumentException("Head dimension mismatch");

        // If this is the first update for this layer, initialize the cache
        if (_cache[layerIdx][0] is null)
        {
            // Clone to avoid modifying the original tensors
            _cache[layerIdx][0] = newKeys.clone();
            _cache[layerIdx][1] = newValues.clone();
            
            // Update cache length (only once, all layers should have same length)
            if (layerIdx == 0)
            {
                CacheLength = (int)newKeys.shape[2];
            }
        }
        else
        {
            // Concatenate new K/V with cached K/V along the sequence dimension
            var oldKeys = _cache[layerIdx][0]!;
            var oldValues = _cache[layerIdx][1]!;
            
            var newCachedKeys = cat(new[] { oldKeys, newKeys }, dim: 2);
            var newCachedValues = cat(new[] { oldValues, newValues }, dim: 2);
            
            // Dispose old cache and update with new concatenated version
            oldKeys.Dispose();
            oldValues.Dispose();
            
            _cache[layerIdx][0] = newCachedKeys;
            _cache[layerIdx][1] = newCachedValues;
            
            // Update cache length (only once, all layers should have same length)
            if (layerIdx == 0)
            {
                CacheLength = (int)newCachedKeys.shape[2];
            }
        }

        // Check if we exceeded max sequence length
        if (CacheLength > _maxSeqLen)
        {
            throw new InvalidOperationException(
                $"Cache length {CacheLength} exceeds maximum {_maxSeqLen}");
        }

        // Return aliases to the cached tensors (do not dispose these!)
        return (_cache[layerIdx][0]!.alias(), _cache[layerIdx][1]!.alias());
    }

    /// <summary>
    /// Gets the cached keys and values for a specific layer.
    /// Returns null if the cache is empty for that layer.
    /// Note: The returned tensors are aliases to the cache, do not dispose them.
    /// </summary>
    /// <param name="layerIdx">Layer index (0-based)</param>
    /// <returns>Tuple of (keys, values) or null if cache is empty</returns>
    public (Tensor keys, Tensor values)? Get(int layerIdx)
    {
        if (layerIdx < 0 || layerIdx >= _nLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIdx));

        if (_cache[layerIdx][0] is null)
            return null;

        return (_cache[layerIdx][0]!.alias(), _cache[layerIdx][1]!.alias());
    }

    /// <summary>
    /// Clears all cached data.
    /// </summary>
    public void Clear()
    {
        for (int i = 0; i < _nLayers; i++)
        {
            _cache[i][0]?.Dispose();
            _cache[i][1]?.Dispose();
            _cache[i][0] = null;
            _cache[i][1] = null;
        }
        
        CacheLength = 0;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            Clear();
            _disposed = true;
        }
    }
}
