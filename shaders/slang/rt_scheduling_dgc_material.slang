import rt_types;
import rt_math;
import rt_scene;
import rt_pbr;

[[vk::constant_id(0)]] const uint SPECIALIZED_ARCHETYPE = 0xFFFFFFFFu;

[[vk::binding(1, 0)]] RWStructuredBuffer<Results> results;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint4> worklist;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> fb;
[[vk::binding(4, 0)]] StructuredBuffer<float> vbuf;
[[vk::binding(5, 0)]] StructuredBuffer<GltfMaterialGpu> matBuf;
[[vk::binding(6, 0)]] StructuredBuffer<uint> triMatBuf;
[[vk::binding(7, 0)]] StructuredBuffer<TextureHeaderGpu> texHeaders;
[[vk::binding(8, 0)]] StructuredBuffer<uint> texPixels;

struct PushConstants {
    uint materialId;
    uint queueCapacity;
    uint dumpRenders;
    uint width;
    uint height;
    uint sceneType;
    uint isGltf;
    uint mode; // 0: Full Render, 1: Pure Shading microbenchmark, 2: Primary Surface Normals / G-Buffer
};

[[vk::push_constant]] PushConstants pc;

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint threadIdx = gid.x;

    // Mode 2 / Specialized 8: Primary Surface Normals (G-Buffer) from compacted queue 0
    if (SPECIALIZED_ARCHETYPE == 8u || (SPECIALIZED_ARCHETYPE != 9u && SPECIALIZED_ARCHETYPE != 10u && pc.mode == 2u)) {
        uint count = wlLoadSnapshot(worklist, 0u);
        if (threadIdx >= count || threadIdx >= pc.queueCapacity) return;

        float4 p = wlLoadRecord(worklist, threadIdx);
        uint pixelIdx = asuint(p.y);
        uint primId = asuint(p.z);
        if (WaveActiveAllEqual(primId)) {
            primId = WaveReadLaneFirst(primId);
        }
        float2 bary = unpackUnorm2x16_spv(asuint(p.w));

        Camera cam = getCamera(pc.sceneType, pc.isGltf);
        uint px_u = pixelIdx % pc.width;
        uint py_u = pixelIdx / pc.width;
        float3 inDir = makePrimaryRayDir(cam, px_u, py_u, pc.width, pc.height);

        float3 normal;
        if (pc.isGltf != 0u) {
            normal = computeSmoothNormal(vbuf, pc.sceneType, 1u, primId, float3(0.0), inDir, bary);
        } else {
            float3 hitPos = (pc.sceneType == SCENE_OUTDOOR_TERRAIN) ? float3(0.0) : (cam.pos + inDir * p.x);
            normal = computeSmoothNormal(vbuf, pc.sceneType, 0u, primId, hitPos, inDir, float2(0.0));
        }
        float3 shadedColor = clamp(normal * 0.5 + 0.5, 0.0, 1.0);
        if (pc.dumpRenders != 0u) {
            fb[pixelIdx] = float4(shadedColor, 1.0);
        } else {
            addWaveLum(results, luminanceToUint(shadedColor));
        }
        return;
    }

    bool isPureShading = (SPECIALIZED_ARCHETYPE != 9u && SPECIALIZED_ARCHETYPE != 10u && pc.mode == 1u);
    uint count = isPureShading ? pc.queueCapacity : wlLoadSnapshot(worklist, pc.materialId);
    if (threadIdx >= count || threadIdx >= pc.queueCapacity) return;

    if (isPureShading && threadIdx == 0u) {
        uint unused;
        InterlockedAdd(results[0].totalRays, count, unused);
        InterlockedAdd(results[0].totalHits, count, unused);
    }

    uint recordIdx = pc.materialId * pc.queueCapacity + threadIdx;
    float4 p = float4(0.0);
    if (!isPureShading) {
        p = wlLoadRecord(worklist, recordIdx);
    }

    uint pixelIdx = isPureShading ? (threadIdx * 8u + (pc.materialId & 7u)) : asuint(p.y);
    if (pixelIdx >= pc.width * pc.height) return;
    uint primId = asuint(p.z);

    Camera cam = getCamera(pc.sceneType, pc.isGltf);
    uint px_u = pixelIdx % pc.width;
    uint py_u = pixelIdx / pc.width;
    float3 inDir = makePrimaryRayDir(cam, px_u, py_u, pc.width, pc.height);

    uint uniformArch = (SPECIALIZED_ARCHETYPE < 8u) ? SPECIALIZED_ARCHETYPE : WaveReadLaneFirst(pc.materialId & 7u);
    float3 shadedColor;

    if (isPureShading) {
        float3 dummyPos = float3(float(px_u) * 0.01, float(py_u) * 0.01, 1.5);
        float3 dummyNorm = ((pixelIdx % 2u) == 0u) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        if (pc.isGltf != 0u) {
            uint matCount, matStride;
            matBuf.GetDimensions(matCount, matStride);
            matCount = max(matCount, 1u);
            GltfMaterialGpu mat = matBuf[uniformArch % matCount];
            mat.archetype = uniformArch;
            float4 dummyTan = float4(1.0, 0.0, 0.0, 1.0);
            float2 dummyUv = float2(float(px_u) / float(pc.width), float(py_u) / float(pc.height));
            shadedColor = evaluateGltfPbrShading(texHeaders, texPixels, SPECIALIZED_ARCHETYPE, mat, dummyPos, dummyNorm, dummyNorm, dummyTan, dummyUv, inDir, pc.sceneType, false, false, float3(1.0, 1.0, -1.0));
        } else {
            shadedColor = evalProceduralMaterialBSDF(pc.sceneType, uniformArch, SPECIALIZED_ARCHETYPE, dummyNorm, inDir, dummyPos, false, false, 1.0, uint3(0xFFFFFFFFu, 0u, 0u));
        }
    } else if (SPECIALIZED_ARCHETYPE == 9u || (SPECIALIZED_ARCHETYPE != 10u && pc.isGltf != 0u)) {
        float2 vis = unpackHalf2x16_spv(asuint(p.x));
        float waterDepth = -1.0;
        if (pc.sceneType == SCENE_AAA_FOREST && uniformArch == 5u) {
            waterDepth = wlLoadRecord(worklist, (8u + pc.materialId) * pc.queueCapacity + threadIdx).x;
        }
        if (WaveActiveAllEqual(primId)) {
            primId = WaveReadLaneFirst(primId);
        }
        uint matId = triMatBuf[primId];
        if (WaveActiveAllEqual(matId)) {
            matId = WaveReadLaneFirst(matId);
        }
        GltfMaterialGpu mat = matBuf[matId];
        mat.archetype = uniformArch;

        float2 bary = unpackUnorm2x16_spv(asuint(p.w));
        float3 interpHitPos, geomNormal, smoothNormal;
        float4 tangent;
        float2 uv;
        unpackGltfHit(vbuf, primId, bary, mat.normalTexIdx >= 0, interpHitPos, geomNormal, smoothNormal, tangent, uv);

        shadedColor = evaluateGltfPbrShading(texHeaders, texPixels, SPECIALIZED_ARCHETYPE, mat, interpHitPos, geomNormal, smoothNormal, tangent, uv, inDir, pc.sceneType, true, false, float3(vis.x, vis.y, waterDepth));
    } else {
        float3 hitPos = cam.pos + inDir * p.x;
        float3 normal = (pc.sceneType == SCENE_OUTDOOR_TERRAIN)
            ? octDecode(unpackSnorm2x16_spv(primId))
            : computeSmoothNormal(vbuf, pc.sceneType, 0u, primId, hitPos, inDir, float2(0.0));
        uint3 aux = uint3(0xFFFFFFFFu, 0u, 0u);
        if (pc.sceneType != SCENE_OUTDOOR_TERRAIN) {
            aux = wlLoadRecordBits(worklist, (8u + pc.materialId) * pc.queueCapacity + threadIdx).xyz;
        }
        shadedColor = evalProceduralMaterialBSDF(pc.sceneType, uniformArch, SPECIALIZED_ARCHETYPE, normal, inDir, hitPos, true, false, p.w, aux);
    }

    if (pc.dumpRenders != 0u) {
        fb[pixelIdx] = float4(tonemapACES(shadedColor), 1.0);
    } else {
        addWaveLum(results, luminanceToUint(shadedColor));
    }
}
