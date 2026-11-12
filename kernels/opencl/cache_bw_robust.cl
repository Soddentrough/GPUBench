__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void run_benchmark(__global const float4* restrict inBuf,
                   __global float* restrict outBuf,
                   __global const uint* restrict pc) {
    const uint lid = get_local_id(0);
    const uint wid = get_group_id(0);
    const uint gid = get_global_id(0);

    const uint stepVec4 = pc ? pc[0] : 1024u;
    const uint n = pc ? pc[1] : 1024u;
    const uint iterations = pc ? pc[2] : 2000u;

    const uint numBlocks = (n >> 10u) > 0u ? (n >> 10u) : 1u;
    uint blockBase = ((wid * 1009u) % numBlocks) << 10u;
    const uint lane = lid;

    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);

    #pragma unroll 1
    for (uint i = 0u; i < iterations; ++i) {
        const uint p0 = blockBase + lane;
        const float4 v0 = inBuf[p0];
        const float4 v1 = inBuf[p0 + 128u];
        const float4 v2 = inBuf[p0 + 256u];
        const float4 v3 = inBuf[p0 + 384u];
        const float4 v4 = inBuf[p0 + 512u];
        const float4 v5 = inBuf[p0 + 640u];
        const float4 v6 = inBuf[p0 + 768u];
        const float4 v7 = inBuf[p0 + 896u];

        acc0 += (v0 + v1) + (v2 + v3);
        acc1 += (v4 + v5) + (v6 + v7);

        blockBase += stepVec4;
        if (blockBase >= n) {
            blockBase -= n;
        }
    }

    const float4 total = acc0 + acc1;
    outBuf[gid & 65535u] = (total.x + total.y) + (total.z + total.w);
}
