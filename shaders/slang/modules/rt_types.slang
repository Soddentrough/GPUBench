module rt_types;

// Vendor-neutral cache-line padded worklist counter indices.
// A stride of 64 uint32_t (256 bytes = 16 uint4) guarantees every queue counter
// resides in its own L2 cache line and distinct L2 bank across AMD/NVIDIA/Intel/Apple.
public static const uint Q_COUNTER_STRIDE = 64u;
public static const uint Q_HEADER_UINT_CAP = 4096u; // 16 KB header
public static const uint Q_HEADER_VEC4_CAP = 1024u; // 4096 uints / 4

// Scene Type Constants
public static const uint SCENE_SHOWROOM_STUDIO = 0u;
public static const uint SCENE_OUTDOOR_TERRAIN = 1u;
public static const uint SCENE_INDOOR_ATRIUM   = 2u;
public static const uint SCENE_AAA_FOREST      = 3u;

// Procedural Showroom Studio & Indoor Atrium Primitive Ranges
public static const uint PRIM_KNOT_PAINT_END  = 1536u;
public static const uint PRIM_KNOT_JADE_END   = 3072u;
public static const uint PRIM_KNOT_CHROME_END = 4608u;
public static const uint PRIM_KNOT_VELVET_END = 6144u;
public static const uint PRIM_KNOT_RUST_END   = 7680u;
public static const uint PRIM_SPHERE_END      = 9600u;
public static const uint PRIM_PEDESTAL_END    = 9728u;
public static const uint PRIM_SUZANNE_END     = 10696u;
public static const uint PRIM_FLOOR_END       = 27080u;
public static const uint PRIM_WALLS_END       = 28104u;
public static const uint PRIM_CEILING_END     = 29128u;
public static const uint PRIM_COLUMNS_END     = 35272u;

// Outdoor Landscape Primitive Ranges
public static const uint PRIM_TERRAIN_END     = 32768u;
public static const uint PRIM_WATER_END       = 34816u;
public static const uint PRIM_TREES_END       = 46016u;

public struct Results {
    public uint totalRays;
    public uint totalHits;
    public uint accumulatedLuminance;
};

public struct VkDispatchIndirectCommand {
    public uint x;
    public uint y;
    public uint z;
};

// Native Vulkan Device-Generated Commands (DGC) sequence token structure (48 bytes)
public struct DGCSequenceItem {
    public uint pipelineIndex;
    public uint pc[8];
    public uint cmdX;
    public uint cmdY;
    public uint cmdZ;
};

public struct GltfMaterialGpu {
    public float4 baseColorFactor;
    public float metallicFactor;
    public float roughnessFactor;
    public float transmissionFactor;
    public float ior;
    public float4 emissiveFactor;
    public int baseColorTexIdx;
    public int metallicRoughnessTexIdx;
    public int normalTexIdx;
    public int occlusionTexIdx;
    public int emissiveTexIdx;
    public float normalScale;
    public float alphaCutoff;
    public uint alphaMode;
    public uint archetype;
    public uint extraParam1;
    public uint extraParam2;
    public uint pad1;
};

public struct TextureHeaderGpu {
    public uint width;
    public uint height;
    public uint offset;
    public uint pad;
};

public struct RayPayload {
    public float3 color;
};

// 128-bit Vectorized WorkList Header & Record Accessors (RWStructuredBuffer<uint4>)
// Live append counter for queue q in [0..15]: uint offset q * 64 -> uint4 index q * 16, .x
public uint wlLoadCounter(RWStructuredBuffer<uint4> wl, uint q) {
    return wl[q * 16u].x;
}

public void wlStoreCounter(RWStructuredBuffer<uint4> wl, uint q, uint val) {
    wl[q * 16u].x = val;
}

public uint wlAtomicAddCounter(RWStructuredBuffer<uint4> wl, uint q, uint val) {
    uint prev;
    InterlockedAdd(wl[q * 16u].x, val, prev);
    return prev;
}

// Resolved snapshot count for queue q in [0..15]: uint offset (16 + q) * 64 -> uint4 index (16 + q) * 16, .x
public uint wlLoadSnapshot(RWStructuredBuffer<uint4> wl, uint q) {
    return wl[(16u + q) * 16u].x;
}

public void wlStoreSnapshot(RWStructuredBuffer<uint4> wl, uint q, uint val) {
    wl[(16u + q) * 16u].x = val;
}

// Contiguous prefix wave sums: uint offset 2048 + q -> uint4 index 512 + (q >> 2), [q & 3]
public uint wlLoadPrefix(RWStructuredBuffer<uint4> wl, uint q) {
    return wl[512u + (q >> 2u)][q & 3u];
}

public void wlStorePrefix(RWStructuredBuffer<uint4> wl, uint q, uint val) {
    wl[512u + (q >> 2u)][q & 3u] = val;
}

// Fast contiguous snapshots: uint offset 2064 + q -> uint4 index 516 + (q >> 2), [q & 3]
public uint wlLoadFastSnap(RWStructuredBuffer<uint4> wl, uint q) {
    return wl[516u + (q >> 2u)][q & 3u];
}

public void wlStoreFastSnap(RWStructuredBuffer<uint4> wl, uint q, uint val) {
    wl[516u + (q >> 2u)][q & 3u] = val;
}

// Per-sample seed & packed metadata slots: uint offset 2080, 2081 -> uint4 index 520, .x/.y
public uint wlLoadSeed(RWStructuredBuffer<uint4> wl) {
    return wl[520u].x;
}

public void wlStoreSeed(RWStructuredBuffer<uint4> wl, uint val) {
    wl[520u].x = val;
}

public uint wlLoadMeta(RWStructuredBuffer<uint4> wl) {
    return wl[520u].y;
}

public void wlStoreMeta(RWStructuredBuffer<uint4> wl, uint val) {
    wl[520u].y = val;
}

// 16-byte Ray Records starting after the 16 KB header (uint4 index Q_HEADER_VEC4_CAP + recIdx)
public float4 wlLoadRecord(RWStructuredBuffer<uint4> wl, uint recIdx) {
    return asfloat(wl[Q_HEADER_VEC4_CAP + recIdx]);
}

public uint4 wlLoadRecordBits(RWStructuredBuffer<uint4> wl, uint recIdx) {
    return wl[Q_HEADER_VEC4_CAP + recIdx];
}

public void wlStoreRecord(RWStructuredBuffer<uint4> wl, uint recIdx, float4 val) {
    wl[Q_HEADER_VEC4_CAP + recIdx] = asuint(val);
}

public void wlStoreRecordBits(RWStructuredBuffer<uint4> wl, uint recIdx, uint4 val) {
    wl[Q_HEADER_VEC4_CAP + recIdx] = val;
}

// Subgroup-coalesced global atomic statistics accumulation
public void addWaveCounters(RWStructuredBuffer<Results> results, uint threadRays, uint threadHits, uint threadLum) {
    uint waveRays = WaveActiveSum(threadRays);
    uint waveHits = WaveActiveSum(threadHits);
    uint waveLum = WaveActiveSum(threadLum);
    if (WaveIsFirstLane()) {
        uint unused;
        if (waveRays > 0u) InterlockedAdd(results[0].totalRays, waveRays, unused);
        if (waveHits > 0u) InterlockedAdd(results[0].totalHits, waveHits, unused);
        if (waveLum > 0u) InterlockedAdd(results[0].accumulatedLuminance, waveLum, unused);
    }
}

public void addWaveHitsLum(RWStructuredBuffer<Results> results, uint threadHits, uint threadLum) {
    uint waveHits = WaveActiveSum(threadHits);
    uint waveLum = WaveActiveSum(threadLum);
    if (WaveIsFirstLane()) {
        uint unused;
        if (waveHits > 0u) InterlockedAdd(results[0].totalHits, waveHits, unused);
        if (waveLum > 0u) InterlockedAdd(results[0].accumulatedLuminance, waveLum, unused);
    }
}

public void addWaveLum(RWStructuredBuffer<Results> results, uint threadLum) {
    uint waveLum = WaveActiveSum(threadLum);
    if (WaveIsFirstLane() && waveLum > 0u) {
        uint unused;
        InterlockedAdd(results[0].accumulatedLuminance, waveLum, unused);
    }
}
