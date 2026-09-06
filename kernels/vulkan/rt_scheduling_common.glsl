#ifndef RT_SCHEDULING_COMMON_GLSL
#define RT_SCHEDULING_COMMON_GLSL

// Vendor-neutral cache-line padded worklist counter indices.
// A stride of 64 uint32_t (256 bytes) guarantees that every queue counter resides
// in its own L2 cache line and maps to a distinct L2 cache bank on AMD RDNA (64B/128B),
// NVIDIA Ampere/Ada/Blackwell (128B), Intel Arc (64B), and Apple Silicon (128B).
// This eliminates cross-CU / cross-wave L2 bank serialization and false sharing.

#define Q_COUNTER_STRIDE   64u
#define Q_HEADER_UINT_CAP  4096u // 16 KB header for cache/page alignment

// Live append counter for queue q (q in [0..15])
#define Q_COUNTER_IDX(q)   ((q) * Q_COUNTER_STRIDE)

// Resolved snapshot count for queue q (q in [0..15])
#define Q_SNAPSHOT_IDX(q)  ((16u + (q)) * Q_COUNTER_STRIDE)

// Prefix wave sum for octant q (q in [0..7])
#define Q_PREFIX_IDX(q)    ((32u + (q)) * Q_COUNTER_STRIDE)

// Native Vulkan Device-Generated Commands (DGC) sequence token structure
struct DGCSequenceItem {
    uint pipelineIndex;
    uint pc[8];
    uint cmdX;
    uint cmdY;
    uint cmdZ;
};

#endif // RT_SCHEDULING_COMMON_GLSL
