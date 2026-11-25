import rt_types;
import rt_math;
import rt_scene;
import rt_pbr;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] RWStructuredBuffer<Results> results;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint4> worklist;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> fb;

struct PushConstants {
    uint queueId;
    uint queueCapacity;
    uint dumpRenders;
    uint width;
    uint height;
    uint sceneType;
    uint isGltf;
    uint lightIndex;
};

[[vk::push_constant]] PushConstants pc;

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint threadIdx = gid.x;
    uint count = min(wlLoadSnapshot(worklist, pc.queueId), pc.queueCapacity);

    uint sampleSeed = 0u;
    if (WaveIsFirstLane()) {
        sampleSeed = wlLoadSeed(worklist);
    }
    sampleSeed = WaveReadLaneFirst(sampleSeed);

    uint threadHits = 0u;
    uint threadLum = 0u;

    if (threadIdx < count) {
        uint recordIdx = pc.queueId * pc.queueCapacity + threadIdx;
        float4 p0 = wlLoadRecord(worklist, recordIdx * 2u);
        uint4 p1 = wlLoadRecordBits(worklist, recordIdx * 2u + 1u);
        float3 hitPos = p0.xyz;
        uint pixelIdx = asuint(p0.w);
        float3 geomNormal = octDecode(unpackSnorm2x16_spv(p1.x));

        float3 lightDir;
        float maxDist;
        getShadowLightParams(pc.sceneType, pc.lightIndex, lightDir, maxDist);

        RayQuery<RAY_FLAG_NONE> sQuery;
        uint shRng = pixelIdx + sampleSeed;
        float vis = traceAreaShadow(topLevelAS, sQuery, hitPos, geomNormal, lightDir, maxDist, 0.045, shRng, 4);
        threadHits = uint(round((1.0 - vis) * 4.0));
        float shadowVal = lerp(0.08, 0.95, vis);

        if (pc.dumpRenders != 0u) {
            fb[pixelIdx] = float4(float3(shadowVal), 1.0);
        } else {
            threadLum = uint(clamp(shadowVal, 0.0, 1.0) * 255.0 + 0.5);
        }
    }

    addWaveHitsLum(results, threadHits, threadLum);
}
