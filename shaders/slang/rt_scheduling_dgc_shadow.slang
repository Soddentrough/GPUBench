import rt_types;
import rt_scene;

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

    uint threadHits = 0u;
    uint threadLum = 0u;

    if (threadIdx < count) {
        uint recordIdx = pc.queueId * pc.queueCapacity + threadIdx;
        float4 p = wlLoadRecord(worklist, recordIdx);
        float3 rayOrig = p.xyz;
        uint pixelIdx = asuint(p.w);

        float3 lightDir;
        float maxDist;
        getShadowLightParams(pc.sceneType, pc.lightIndex, lightDir, maxDist);

        bool occluded = false;
        {
            RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE> sQuery;
            RayDesc ray = { rayOrig, 0.05, lightDir, maxDist };
            sQuery.TraceRayInline(topLevelAS, 0u, 0xFFu, ray);
            while (sQuery.Proceed()) {}
            occluded = (sQuery.CommittedStatus() != COMMITTED_NOTHING);
        }

        if (occluded) {
            threadHits = 1u;
        }

        if (pc.dumpRenders != 0u) {
            float shadowVal = occluded ? 0.08 : 0.95;
            fb[pixelIdx] = float4(float3(shadowVal), 1.0);
        } else {
            threadLum = occluded ? 20u : 243u;
        }
    }

    addWaveHitsLum(results, threadHits, threadLum);
}
