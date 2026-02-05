using NanoChat.Core.Model;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace NanoChat.Core.Tests;

public class KVCacheTests
{
    [Fact]
    public void Constructor_InitializesEmptyCache()
    {
        using var cache = new KVCache(nLayers: 4, nHeads: 8, headDim: 64, batchSize: 2, maxSeqLen: 512);
        
        Assert.Equal(0, cache.CacheLength);
    }

    [Fact]
    public void Update_FirstUpdate_InitializesCache()
    {
        using var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 128);
        
        // Create initial K/V tensors: (batch=1, nHead=4, seqLen=5, headDim=32)
        using var keys = randn(1, 4, 5, 32);
        using var values = randn(1, 4, 5, 32);
        
        // Update layer 0 (returns aliases, don't dispose)
        var (cachedK, cachedV) = cache.Update(0, keys, values);
        
        Assert.Equal(5, cache.CacheLength);
        Assert.Equal(new long[] { 1, 4, 5, 32 }, cachedK.shape);
        Assert.Equal(new long[] { 1, 4, 5, 32 }, cachedV.shape);
    }

    [Fact]
    public void Update_SubsequentUpdate_ConcatenatesCache()
    {
        using var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 128);
        
        // First update with 5 tokens (aliases, don't dispose)
        using var keys1 = randn(1, 4, 5, 32);
        using var values1 = randn(1, 4, 5, 32);
        var (cached1K, cached1V) = cache.Update(0, keys1, values1);
        
        Assert.Equal(5, cache.CacheLength);
        
        // Second update with 3 more tokens (aliases, don't dispose)
        using var keys2 = randn(1, 4, 3, 32);
        using var values2 = randn(1, 4, 3, 32);
        var (cached2K, cached2V) = cache.Update(0, keys2, values2);
        
        Assert.Equal(8, cache.CacheLength); // 5 + 3
        Assert.Equal(new long[] { 1, 4, 8, 32 }, cached2K.shape);
        Assert.Equal(new long[] { 1, 4, 8, 32 }, cached2V.shape);
    }

    [Fact]
    public void Update_MultipleTokens_Works()
    {
        using var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 128);
        
        // Update with multiple tokens at once (aliases, don't dispose)
        using var keys = randn(1, 4, 10, 32);
        using var values = randn(1, 4, 10, 32);
        var (cachedK, cachedV) = cache.Update(0, keys, values);
        
        Assert.Equal(10, cache.CacheLength);
    }

    [Fact]
    public void Update_MultipleLayers_IndependentCaches()
    {
        using var cache = new KVCache(nLayers: 3, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 128);
        
        // Update each layer with different values (aliases, don't dispose)
        for (int layer = 0; layer < 3; layer++)
        {
            using var keys = ones(1, 4, 5, 32) * (layer + 1);
            using var values = ones(1, 4, 5, 32) * (layer + 1);
            var (cachedK, cachedV) = cache.Update(layer, keys, values);
        }
        
        // Verify each layer has independent cache
        for (int layer = 0; layer < 3; layer++)
        {
            var result = cache.Get(layer);
            Assert.NotNull(result);
            
            var (cachedK, cachedV) = result.Value;
            Assert.Equal(new long[] { 1, 4, 5, 32 }, cachedK.shape);
            
            // Verify the values match what we stored
            var expectedValue = layer + 1;
            var actualValue = cachedK[0, 0, 0, 0].item<float>();
            Assert.Equal(expectedValue, actualValue, precision: 5);
        }
    }

    [Fact]
    public void Update_BatchSize_Validated()
    {
        using var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 2, maxSeqLen: 128);
        
        // Wrong batch size
        using var keys = randn(1, 4, 5, 32); // batch=1 instead of 2
        using var values = randn(1, 4, 5, 32);
        
        Assert.Throws<ArgumentException>(() => cache.Update(0, keys, values));
    }

    [Fact]
    public void Update_NumHeads_Validated()
    {
        using var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 128);
        
        // Wrong number of heads
        using var keys = randn(1, 8, 5, 32); // nHead=8 instead of 4
        using var values = randn(1, 8, 5, 32);
        
        Assert.Throws<ArgumentException>(() => cache.Update(0, keys, values));
    }

    [Fact]
    public void Update_HeadDim_Validated()
    {
        using var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 128);
        
        // Wrong head dimension
        using var keys = randn(1, 4, 5, 64); // headDim=64 instead of 32
        using var values = randn(1, 4, 5, 64);
        
        Assert.Throws<ArgumentException>(() => cache.Update(0, keys, values));
    }

    [Fact]
    public void Update_ExceedsMaxSeqLen_Throws()
    {
        using var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 10);
        
        // First update with 8 tokens (aliases, don't dispose)
        using var keys1 = randn(1, 4, 8, 32);
        using var values1 = randn(1, 4, 8, 32);
        var (cached1K, cached1V) = cache.Update(0, keys1, values1);
        
        // Second update with 5 more tokens (total 13 > max 10)
        using var keys2 = randn(1, 4, 5, 32);
        using var values2 = randn(1, 4, 5, 32);
        
        Assert.Throws<InvalidOperationException>(() => cache.Update(0, keys2, values2));
    }

    [Fact]
    public void Get_EmptyCache_ReturnsNull()
    {
        using var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 128);
        
        var result = cache.Get(0);
        Assert.Null(result);
    }

    [Fact]
    public void Get_PopulatedCache_ReturnsData()
    {
        using var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 128);
        
        using var keys = randn(1, 4, 5, 32);
        using var values = randn(1, 4, 5, 32);
        var (updatedK, updatedV) = cache.Update(0, keys, values);
        updatedK.Dispose();
        updatedV.Dispose();
        
        var result = cache.Get(0);
        Assert.NotNull(result);
        
        var (cachedK, cachedV) = result.Value;
        Assert.Equal(new long[] { 1, 4, 5, 32 }, cachedK.shape);
        Assert.Equal(new long[] { 1, 4, 5, 32 }, cachedV.shape);
    }

    [Fact]
    public void Clear_RemovesAllCachedData()
    {
        using var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 128);
        
        // Populate cache (aliases, don't dispose)
        using var keys = randn(1, 4, 5, 32);
        using var values = randn(1, 4, 5, 32);
        var (cachedK, cachedV) = cache.Update(0, keys, values);
        
        Assert.Equal(5, cache.CacheLength);
        
        // Clear cache
        cache.Clear();
        
        Assert.Equal(0, cache.CacheLength);
        Assert.Null(cache.Get(0));
    }

    [Fact]
    public void Update_InvalidLayerIndex_Throws()
    {
        using var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 128);
        
        using var keys = randn(1, 4, 5, 32);
        using var values = randn(1, 4, 5, 32);
        
        // Negative layer index
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Update(-1, keys, values));
        
        // Layer index >= nLayers
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Update(2, keys, values));
    }

    [Fact]
    public void Get_InvalidLayerIndex_Throws()
    {
        using var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 128);
        
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Get(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Get(2));
    }

    [Fact]
    public void Dispose_CleansUpResources()
    {
        var cache = new KVCache(nLayers: 2, nHeads: 4, headDim: 32, batchSize: 1, maxSeqLen: 128);
        
        // Populate cache (aliases, don't dispose)
        using var keys = randn(1, 4, 5, 32);
        using var values = randn(1, 4, 5, 32);
        var (cachedK, cachedV) = cache.Update(0, keys, values);
        
        cache.Dispose();
        
        // Should be safe to call multiple times
        cache.Dispose();
    }
}
