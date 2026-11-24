import rt_types;
import rt_math;
import rt_scene;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] RWStructuredBuffer<Results> results;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> fb;

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

[shader("raygeneration")]
void main() {
    uint idx = DispatchRaysIndex().x;
    if (idx >= pc.rayCount) return;

    Camera cam = getCamera(pc.sceneType, pc.isGltf);
    uint px_u, py_u;
    computePixelCoords(idx, pc.spatialPattern, pc.width, pc.height, pc.isGltf, px_u, py_u);
    uint pixelIdx = py_u * pc.width + px_u;

    float3 rayDir = makePrimaryRayDir(cam, px_u, py_u, pc.width, pc.height);
    float tMinPrimary, tMaxPrimary;
    getPrimaryRayRange(pc.sceneType, pc.isGltf, tMinPrimary, tMaxPrimary);

    RayDesc ray = { cam.pos, tMinPrimary, rayDir, tMaxPrimary };
    RayPayload p = { float3(0.0) };
    TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, 0u, 0u, 0u, ray, p);

    bool isHit = (p.color.x > 0.0);
    float3 shadedCol = isHit ? p.color : float3(0.0);

    if (pc.dumpRenders != 0u) {
        if (pc.mode == 7u) {
            float3 outCol = isHit ? shadedCol : float3(0.04, 0.05, 0.08);
            fb[pixelIdx] = float4(outCol, 1.0);
        } else {
            fb[pixelIdx] = float4(tonemapACES(shadedCol), 1.0);
        }
    }

    uint waveRays = WaveActiveSum(1u);
    uint waveHits = WaveActiveSum(isHit ? 1u : 0u);
    uint lumVal = isHit ? (uint(clamp(dot(shadedCol, float3(0.2126, 0.7152, 0.0722)), 0.0, 1000.0) * 256.0) + 1u) : 0u;
    uint waveLum = WaveActiveSum(lumVal);
    if (WaveIsFirstLane()) {
        uint unused;
        if (waveRays > 0u) InterlockedAdd(results[0].totalRays, waveRays, unused);
        if (waveHits > 0u) InterlockedAdd(results[0].totalHits, waveHits, unused);
        if (pc.dumpRenders == 0u && waveLum > 0u) InterlockedAdd(results[0].accumulatedLuminance, waveLum, unused);
    }
}
