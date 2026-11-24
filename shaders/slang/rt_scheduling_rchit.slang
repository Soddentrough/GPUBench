import rt_types;
import rt_math;
import rt_scene;
import rt_pbr;

[[vk::constant_id(0)]] const uint RCHIT_MODE = 7u;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] RWStructuredBuffer<Results> results;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> fb;
[[vk::binding(3, 0)]] StructuredBuffer<float> vbuf;
[[vk::binding(4, 0)]] StructuredBuffer<GltfMaterialGpu> matBuf;
[[vk::binding(5, 0)]] StructuredBuffer<uint> triMatBuf;
[[vk::binding(6, 0)]] StructuredBuffer<TextureHeaderGpu> texHeaders;
[[vk::binding(7, 0)]] StructuredBuffer<uint> texPixels;

struct PushConstants {
    uint rayCount;
    uint mode;
    uint bounces;
    uint seed;
    uint dumpRenders;
    uint width;
    uint height;
    uint spatialPattern;
    uint sceneType;
    uint isGltf;
    uint spp;
    uint sampleIdx;
};

[[vk::push_constant]] PushConstants pc;

[shader("closesthit")]
void main(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    uint primId = PrimitiveIndex();
    float t = RayTCurrent();
    float3 rayOrigin = WorldRayOrigin();
    float3 rayDir = WorldRayDirection();
    float3 hitPos = rayOrigin + rayDir * t;
    float2 rawBary = attr.barycentrics;

    if (RCHIT_MODE == 7u || pc.mode == 7u) {
        float3 normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, primId, hitPos, rayDir, rawBary);
        payload.color = max(normal * 0.5 + 0.5, float3(1e-4));
        return;
    }

    if (RCHIT_MODE != 7u) {
        uint rngState = primId + pc.seed;
        float3 pixelColor = float3(0.0);

        if (pc.mode == 4u) {
            uint idx = DispatchRaysIndex().x;
            uint matId = idx % 8u;
            uint px_u = idx % pc.width;
            uint py_u = idx / pc.width;
            float3 dummyPos = float3(float(px_u) * 0.01, float(py_u) * 0.01, 1.5);
            float3 dummyNorm = ((idx % 2u) == 0u) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
            rngState = idx + pc.seed;
            if (pc.isGltf != 0u) {
                uint matCount, matStride;
                matBuf.GetDimensions(matCount, matStride);
                matCount = max(matCount, 1u);
                GltfMaterialGpu mat = matBuf[matId % matCount];
                mat.archetype = matId;
                float4 dummyTan = float4(1.0, 0.0, 0.0, 1.0);
                float2 dummyUv = float2(float(px_u) / float(pc.width), float(py_u) / float(pc.height));
                pixelColor = evaluateGltfPbrShading(texHeaders, texPixels, 0xFFFFFFFFu, mat, dummyPos, dummyNorm, dummyNorm, dummyTan, dummyUv, rayDir, pc.sceneType, false, false, float3(1.0, 1.0, -1.0));
            } else {
                float3 sunDir, sunColor;
                getSceneSunParams(pc.sceneType, sunDir, sunColor);
                float3 baseColor = getSceneArchetypeAlbedo(pc.sceneType, matId);
                pixelColor = evaluateMaterialArchetypeDirect(matId, 0xFFFFFFFFu, dummyPos, baseColor, 0.5, 0.0, 1.5, dummyNorm, -rayDir, sunDir, sunColor, 1.0);
            }
            payload.color = max(pixelColor, float3(1e-4));
            return;
        }

        if (pc.isGltf != 0u) {
            float2 bary = unpackUnorm2x16_spv(packUnorm2x16_spv(rawBary));
            float3 interpHitPos, geomNormal, smoothNormal;
            float4 tangent;
            float2 uv;
            unpackGltfHit(vbuf, primId, bary, true, interpHitPos, geomNormal, smoothNormal, tangent, uv);
            uint matId = triMatBuf[primId];
            GltfMaterialGpu mat = matBuf[matId];
            RayQuery<RAY_FLAG_NONE> rq;
            pixelColor = evaluateGltfPbrWithRayQuery(topLevelAS, rq, texHeaders, texPixels, 0xFFFFFFFFu, mat, interpHitPos, geomNormal, smoothNormal, tangent, uv, rayDir, pc.sceneType, true, false, rngState);
        } else {
            float3 normal = computeSmoothNormal(vbuf, pc.sceneType, 0u, primId, hitPos, rayDir, rawBary);
            uint matId = computeMaterialId(pc.sceneType, primId, hitPos, normal);
            float3 sunDir, sunColor;
            getSceneSunParams(pc.sceneType, sunDir, sunColor);
            float3 baseColor = getSceneArchetypeAlbedo(pc.sceneType, matId);
            pixelColor = evaluateMaterialArchetypeDirect(matId, 0xFFFFFFFFu, hitPos, baseColor, 0.5, 0.0, 1.5, normal, -rayDir, sunDir, sunColor, 1.0);
        }

        payload.color = max(pixelColor, float3(1e-4));
    }
}
