import rt_types;
import rt_math;
import rt_scene;
import rt_pbr;

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
    uint binIndex;
};

[[vk::push_constant]] PushConstants pc;

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint threadIdx = gid.x;
    uint count = min(wlLoadSnapshot(worklist, pc.queueId), pc.queueCapacity);

    uint threadLum = 0u;

    if (threadIdx < count) {
        uint recordIdx = pc.queueId * pc.queueCapacity + threadIdx;
        uint4 raw = wlLoadRecordBits(worklist, recordIdx);
        float3 smoothNormal = octDecode(unpackSnorm2x16_spv(raw.x));
        uint pixelIdx = raw.y & 0x00FFFFFFu;
        float metallic = float(raw.y >> 24u) * (1.0 / 255.0);
        float2 albRG = unpackHalf2x16_spv(raw.z);
        float2 albBR = unpackHalf2x16_spv(raw.w);
        float3 albedo = float3(albRG, albBR.x);
        float roughness = albBR.y;

        Camera cam = getCamera(pc.sceneType, pc.isGltf);
        uint px_u = pixelIdx % pc.width;
        uint py_u = pixelIdx / pc.width;
        float3 inDir = makePrimaryRayDir(cam, px_u, py_u, pc.width, pc.height);
        float3 viewDir = -inDir;

        float3 outRadiance;
        if (pc.binIndex <= 1u || pc.binIndex == 0xFFFFFFFFu) {
            float3 L, C;
            getMultiLight(0u, 1u, pc.sceneType, L, C);
            outRadiance = evalCookTorranceDirect(smoothNormal, viewDir, L, C, albedo, roughness, metallic);
        } else {
            float3 totalRadiance = float3(0.0);
            uint lane16 = WaveGetLaneIndex() & 15u;
            uint qOct = WaveReadLaneFirst(pc.queueId & 7u);
            uint oppositeOctant = (~qOct) & 7u;
            bool has16 = ((WaveActiveBallot(true).x & 0xFFFFu) == 0xFFFFu);

            for (uint b = 0u; b < 8u; ++b) {
                if (b == oppositeOctant) continue;
                float3 octSign = float3(
                    (b & 1u) != 0u ? 1.0 : -1.0,
                    (b & 2u) != 0u ? 1.0 : -1.0,
                    (b & 4u) != 0u ? 1.0 : -1.0
                );
                float3 s = smoothNormal * octSign;
                float maxDot = length(max(s, float3(0.0))) + min(s.x, 0.0) * 0.06 + min(s.y, 0.0) * 0.06 + min(s.z, 0.0) * 0.20;
                if (WaveActiveAllTrue(maxDot <= 0.0)) continue;

                if (has16) {
                    float3 binL, binC;
                    getMultiLight(b * 16u + lane16, 128u, pc.sceneType, binL, binC);
                    for (uint li = 0u; li < 16u; ++li) {
                        float3 L = WaveReadLaneAt(binL, li);
                        float3 C = WaveReadLaneAt(binC, li);
                        float NdotL = dot(smoothNormal, L);
                        if (WaveActiveAllTrue(NdotL <= 0.0)) continue;
                        if (NdotL > 0.0) {
                            totalRadiance += evalCookTorranceDirect(smoothNormal, viewDir, L, C, albedo, roughness, metallic);
                        }
                    }
                } else {
                    for (uint li = 0u; li < 16u; ++li) {
                        float3 L, C;
                        getMultiLight(b * 16u + li, 128u, pc.sceneType, L, C);
                        if (dot(smoothNormal, L) > 0.0) {
                            totalRadiance += evalCookTorranceDirect(smoothNormal, viewDir, L, C, albedo, roughness, metallic);
                        }
                    }
                }
            }
            outRadiance = totalRadiance;
        }

        if (pc.dumpRenders != 0u) {
            fb[pixelIdx] = float4(tonemapACES(outRadiance), 1.0);
        } else {
            threadLum = luminanceToUint(outRadiance);
        }
    }

    addWaveLum(results, threadLum);
}
