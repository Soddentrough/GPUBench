import rt_types;
import rt_math;
import rt_scene;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] RWStructuredBuffer<Results> results;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> fb;
[[vk::binding(3, 0)]] StructuredBuffer<float> vbuf;
[[vk::binding(4, 0)]] StructuredBuffer<GltfMaterialGpu> matBuf;
[[vk::binding(5, 0)]] StructuredBuffer<uint> triMatBuf;

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
        uint rngState = pixelIdx + pc.seed;
        uint totalSpp = max(1u, pc.spp);
        float2 jitter = float2(0.0, 0.0);
        if (totalSpp > 1u) {
            jitter.x = randFloat(rngState) - 0.5;
            jitter.y = randFloat(rngState) - 0.5;
        }
        float3 curOrigin = rayOrigin;
        float3 curDir = (totalSpp > 1u)
            ? makePrimaryRayDir(cam, px_u, py_u, pc.width, pc.height, jitter)
            : rayDir;
        float3 throughput = float3(1.0);
        uint totalBounces = max(1u, pc.bounces);
        float surfEps = (pc.isGltf != 0u) ? 0.05 : 0.005;

        for (uint b = 0u; b < totalBounces; ++b) {
            myRays++;
            RayPayload p = { float3(0.0) };
            RayDesc ray = { curOrigin, (b == 0u) ? tMinPrimary : surfEps, curDir, tMaxPrimary };
            HitObject hObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, 0u, 0u, 0u, ray, p);
            ReorderThread(hObj);
            HitObject::Invoke(topLevelAS, hObj, p);

            if (hObj.IsHit()) {
                myHits++;
                pixelColor += throughput * max(p.color, float3(0.0));

                if (b + 1u < totalBounces) {
                    if (b >= 2u && randFloat(rngState) < 0.4) break;

                    float tHit = hObj.GetRayDesc().TMax;
                    uint primId = hObj.GetPrimitiveIndex();
                    float2 rawBary = hObj.GetAttributes<BuiltInTriangleIntersectionAttributes>().barycentrics;
                    float3 hitPos = curOrigin + curDir * tHit;
                    float3 surfNormal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, primId, hitPos, curDir, rawBary);

                    float3 surfAlbedo = float3(0.72);
                    float surfMetal = 0.0;
                    float surfRough = 0.5;
                    if (pc.isGltf != 0u) {
                        uint matId = triMatBuf[primId];
                        GltfMaterialGpu mat = matBuf[matId];
                        if (mat.transmissionFactor > 0.5 || mat.archetype == 2u) break;
                        surfAlbedo = mat.baseColorFactor.rgb;
                        surfMetal = clamp(mat.metallicFactor, 0.0, 1.0);
                        surfRough = clamp(mat.roughnessFactor, 0.04, 1.0);
                    } else {
                        uint arch = computeMaterialId(pc.sceneType, primId, hitPos, surfNormal);
                        surfAlbedo = getSceneArchetypeAlbedo(pc.sceneType, arch);
                    }

                    float u1 = randFloat(rngState);
                    float u2 = randFloat(rngState);
                    if (surfMetal > 0.1 && u2 < surfMetal) {
                        float3 refl = reflect(curDir, surfNormal);
                        float3 jitterVec = (float3(u1, u2, frac(u1 * 31.7)) * 2.0 - 1.0) * (surfRough * 0.4);
                        curDir = normalize(refl + jitterVec);
                        if (dot(curDir, surfNormal) <= 0.0) curDir = refl;
                        throughput *= lerp(float3(0.04), surfAlbedo, surfMetal);
                    } else {
                        curDir = alignToNormal(cosineSampleHemisphere(u1, u2), surfNormal);
                        throughput *= (1.0 - surfMetal) * surfAlbedo;
                    }
                    curOrigin = hitPos + surfNormal * surfEps;
                }
            } else {
                pixelColor += throughput * abs(p.color);
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
            uint primId = hObj.GetPrimitiveIndex();
            float2 rawBary = hObj.GetAttributes<BuiltInTriangleIntersectionAttributes>().barycentrics;
            float3 hitPos = rayOrigin + rayDir * tHit;
            float3 surfNormal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, primId, hitPos, rayDir, rawBary);

            uint rngState = pixelIdx + pc.seed;
            float u1 = randFloat(rngState);
            float u2 = randFloat(rngState);
            float3 scatterDir = alignToNormal(cosineSampleHemisphere(u1, u2), surfNormal);

            myRays++;
            RayPayload gp = { float3(0.0) };
            float surfEps = (pc.isGltf != 0u) ? 0.05 : 0.005;
            RayDesc gRay = { hitPos + surfNormal * surfEps, 0.01, scatterDir, 100.0 };
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
        } else if (pc.mode == 1u) {
            uint totalSpp = max(1u, pc.spp);
            uint sampleIdx = pc.sampleIdx;
            float expGain = (pc.sceneType == SCENE_INDOOR_ATRIUM) ? 1.6 : 1.0;
            if (totalSpp <= 1u) {
                fb[pixelIdx] = float4(tonemapACES(pixelColor * expGain), 1.0);
            } else if (sampleIdx == 0u) {
                fb[pixelIdx] = float4(pixelColor, 1.0);
            } else if (sampleIdx < totalSpp - 1u) {
                fb[pixelIdx] = fb[pixelIdx] + float4(pixelColor, 0.0);
            } else {
                float3 total = fb[pixelIdx].rgb + pixelColor;
                float3 avg = (total / float(totalSpp)) * expGain;
                fb[pixelIdx] = float4(tonemapACES(avg), 1.0);
            }
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
