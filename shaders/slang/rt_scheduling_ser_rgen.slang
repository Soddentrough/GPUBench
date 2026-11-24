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
    float3 rayOrigin = cam.pos;
    float tMinPrimary, tMaxPrimary;
    getPrimaryRayRange(pc.sceneType, pc.isGltf, tMinPrimary, tMaxPrimary);

    uint myRays = 0u;
    uint myHits = 0u;
    float3 pixelColor = float3(0.0);

    if (pc.mode == 1u) {
        // Multi-Bounce Path Tracing with SER
        float3 curOrigin = rayOrigin;
        float3 curDir = rayDir;
        float3 throughput = float3(1.0);
        uint totalBounces = max(1u, pc.bounces);
        uint rngState = pixelIdx + pc.seed;

        for (uint b = 0u; b < totalBounces; ++b) {
            myRays++;
            RayPayload p = { float3(0.0) };
            RayDesc ray = { curOrigin, (b == 0u) ? tMinPrimary : 0.05, curDir, tMaxPrimary };
            HitObject hObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, 0u, 0u, 0u, ray, p);
            ReorderThread(hObj);

            if (hObj.IsHit()) {
                myHits++;
                HitObject::Invoke(topLevelAS, hObj, p);
                pixelColor += throughput * p.color;

                if (b >= 2u) {
                    float survP = max(throughput.r, max(throughput.g, throughput.b));
                    if (float(serHash(rngState + b) % 1000u) / 1000.0 > survP) break;
                    throughput /= max(survP, 0.01);
                }

                float tHit = hObj.GetRayDesc().TMax;
                curOrigin = curOrigin + curDir * tHit;
                uint s1 = serHash(rngState + b * 2u);
                uint s2 = serHash(s1);
                float u1 = float(s1 % 10000u) / 10000.0;
                float u2 = float(s2 % 10000u) / 10000.0;
                float theta = 6.28318530718 * u1;
                float phi = acos(clamp(1.0 - 2.0 * u2, -1.0, 1.0));
                curDir = normalize(float3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)));
                throughput *= 0.7;
            } else {
                break;
            }
        }
    } else if (pc.mode == 5u) {
        // Ray-Traced Shadow Rays with SER
        myRays++;
        RayPayload p = { float3(0.0) };
        RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
        HitObject hObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, 0u, 0u, 0u, ray, p);
        ReorderThread(hObj);

        if (hObj.IsHit()) {
            myHits++;
            HitObject::Invoke(topLevelAS, hObj, p);
            float tHit = hObj.GetRayDesc().TMax;
            float3 hitPos = rayOrigin + rayDir * tHit;
            float3 sunDir = normalize(float3(0.45, 0.35, 0.82));
            if (pc.sceneType == SCENE_INDOOR_ATRIUM) {
                sunDir = normalize(float3(0.35, 0.90, -0.25));
            } else if (pc.sceneType == SCENE_SHOWROOM_STUDIO) {
                sunDir = normalize(float3(0.6, 0.8, 0.5));
            }

            myRays++;
            RayPayload sp = { float3(0.0) };
            RayDesc sRay = { hitPos + sunDir * 0.05, 0.01, sunDir, 2500.0 };
            HitObject sObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFFu, 0u, 0u, 0u, sRay, sp);
            ReorderThread(sObj);
            bool shadowed = sObj.IsHit();
            pixelColor = shadowed ? float3(0.1) : p.color;
        }
    } else if (pc.mode == 2u) {
        // Incoherent Secondary Diffuse GI with SER
        myRays++;
        RayPayload p = { float3(0.0) };
        RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
        HitObject hObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, 0u, 0u, 0u, ray, p);
        ReorderThread(hObj);

        if (hObj.IsHit()) {
            myHits++;
            HitObject::Invoke(topLevelAS, hObj, p);
            float tHit = hObj.GetRayDesc().TMax;
            float3 hitPos = rayOrigin + rayDir * tHit;

            uint s1 = serHash(pixelIdx + pc.seed);
            uint s2 = serHash(s1);
            float u1 = float(s1 % 10000u) / 10000.0;
            float u2 = float(s2 % 10000u) / 10000.0;
            float theta = 6.28318530718 * u1;
            float phi = acos(clamp(1.0 - 2.0 * u2, -1.0, 1.0));
            float3 scatterDir = normalize(float3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)));

            myRays++;
            RayPayload gp = { float3(0.0) };
            RayDesc gRay = { hitPos + scatterDir * 0.05, 0.01, scatterDir, 100.0 };
            HitObject gObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, 0u, 0u, 0u, gRay, gp);
            ReorderThread(gObj);
            if (gObj.IsHit()) {
                myHits++;
                HitObject::Invoke(topLevelAS, gObj, gp);
            }
            pixelColor = gp.color;
        }
    } else {
        // Mode 0, 4 & 7: Full Frame, Material Shading (SER coherence hint), or Primary Rays G-Buffer
        myRays++;
        RayPayload p = { float3(0.0) };
        RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
        HitObject hObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, 0u, 0u, 0u, ray, p);
        if (pc.mode == 4u) {
            uint matHint = idx % 8u;
            ReorderThread(hObj, matHint, 3u);
        } else {
            ReorderThread(hObj);
        }
        HitObject::Invoke(topLevelAS, hObj, p);
        if (pc.mode == 4u || hObj.IsHit()) myHits++;
        pixelColor = abs(p.color);
    }

    if (pc.dumpRenders != 0u) {
        if (pc.mode == 7u) {
            float3 outCol = (myHits > 0u) ? pixelColor : float3(0.04, 0.05, 0.08);
            fb[pixelIdx] = float4(outCol, 1.0);
        } else {
            fb[pixelIdx] = float4(tonemapACES(pixelColor), 1.0);
        }
    }

    uint waveRays = WaveActiveSum(myRays);
    uint waveHits = WaveActiveSum(myHits);
    uint lumVal = uint(clamp(dot(pixelColor, float3(0.2126, 0.7152, 0.0722)), 0.0, 1000.0) * 256.0) + 1u;
    uint waveLum = WaveActiveSum(lumVal);
    if (WaveIsFirstLane()) {
        uint unused;
        if (waveRays > 0u) InterlockedAdd(results[0].totalRays, waveRays, unused);
        if (waveHits > 0u) InterlockedAdd(results[0].totalHits, waveHits, unused);
        if (pc.dumpRenders == 0u && waveLum > 0u) InterlockedAdd(results[0].accumulatedLuminance, waveLum, unused);
    }
}
