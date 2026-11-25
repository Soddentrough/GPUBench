import rt_types;
import rt_math;
import rt_scene;
import rt_pbr;

[[vk::constant_id(0)]] const uint MEGA_MODE = 0xFFFFFFFFu;

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
    uint mode; // 0..12
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

[numthreads(8, 4, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint gidx : SV_GroupIndex, uint3 wgid : SV_GroupID) {
    const uint mode = (MEGA_MODE != 0xFFFFFFFFu) ? MEGA_MODE : pc.mode;
    uint3 nwg = WorkgroupCount();
    uint idx = (nwg.y > 1u) ? (gid.y * pc.width + gid.x) : (wgid.x * 32u + gidx);
    if (idx >= pc.rayCount) return;

    uint threadRays = 0u;
    uint threadHits = 0u;
    uint threadLum = 0u;

    Camera cam = getCamera(pc.sceneType, pc.isGltf);
    uint tileIdx = (nwg.y > 1u) ? (idx >> 5u) : wgid.x;
    uint inTile = (nwg.y > 1u) ? (idx & 31u) : gidx;
    uint px_u, py_u;
    computePixelCoords(idx, tileIdx, inTile, pc.spatialPattern, pc.width, pc.height, pc.isGltf, px_u, py_u);
    uint pixelIdx = py_u * pc.width + px_u;
    uint rngState = pixelIdx + pc.seed;

    float3 rayDir = makePrimaryRayDir(cam, px_u, py_u, pc.width, pc.height);
    float3 rayOrigin = cam.pos;
    float tMinPrimary, tMaxPrimary;
    getPrimaryRayRange(pc.sceneType, pc.isGltf, tMinPrimary, tMaxPrimary);

    RayQuery<RAY_FLAG_NONE> rq;

    if (mode == 1u) {
        // Multi-Bounce Path Tracing with Russian Roulette & Progressive Radiance Accumulation
        uint sampleIdx = pc.sampleIdx;
        uint totalSpp = max(1u, pc.spp);
        uint ptRng = pixelIdx + pc.seed;
        float2 jitter = float2(0.0, 0.0);
        if (totalSpp != 1u) {
            jitter.x = randFloat(ptRng) - 0.5;
            jitter.y = randFloat(ptRng) - 0.5;
        }
        float3 curDir = makePrimaryRayDir(cam, px_u, py_u, pc.width, pc.height, jitter);
        float3 curOrigin = cam.pos;

        threadRays++;
        bool hitFound = false;
        float t = 0.0;
        uint primId = 0u;
        float2 rawBary = float2(0.0);
        {
            RayDesc ray = { curOrigin, tMinPrimary, curDir, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                hitFound = true;
                t = rq.CommittedRayT();
                primId = rq.CommittedPrimitiveIndex();
                rawBary = rq.CommittedTriangleBarycentrics();
            }
        }

        float3 sampleRadiance = float3(0.0);
        if (hitFound) {
            threadHits++;
            float3 hitPos = curOrigin + curDir * t;
            float3 normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, primId, hitPos, curDir, rawBary);

            float u1 = randFloat(ptRng);
            float u2 = randFloat(ptRng);
            float3 nextDir = alignToNormal(cosineSampleHemisphere(u1, u2), normal);

            float3 albedo = float3(0.72);
            float3 emissive = float3(0.0);
            float metallic = 0.0;
            float roughness = 0.5;
            float transmission = 0.0;
            uint archetype = 0u;

            if (pc.isGltf != 0u) {
                float2 bary = unpackUnorm2x16_spv(packUnorm2x16_spv(rawBary));
                float2 uv = unpackGltfUV(vbuf, primId, bary);
                uint matId = triMatBuf[primId];
                GltfMaterialGpu mat = matBuf[matId];
                albedo = mat.baseColorFactor.rgb;
                if (mat.baseColorTexIdx >= 0) {
                    albedo *= pow(sampleTexture(texHeaders, texPixels, uint(mat.baseColorTexIdx), uv).rgb, float3(2.2));
                }
                if (mat.emissiveTexIdx >= 0) {
                    emissive = pow(sampleTexture(texHeaders, texPixels, uint(mat.emissiveTexIdx), uv).rgb, float3(2.2)) * mat.emissiveFactor.rgb;
                }
                metallic = mat.metallicFactor;
                roughness = mat.roughnessFactor;
                transmission = mat.transmissionFactor;
                archetype = mat.archetype;
            } else {
                uint matId = computeMaterialId(pc.sceneType, primId, hitPos, normal);
                archetype = matId;
                albedo = getSceneArchetypeAlbedo(pc.sceneType, archetype);
            }

            float3 directLight = float3(0.0);
            float3 throughput = albedo;
            float rayOffset = (pc.isGltf != 0u) ? 0.05 : 0.005;
            float3 bOrigin = hitPos + normal * rayOffset;
            float3 bDir = nextDir;
            bool terminatePath = false;

            if (transmission > 0.5 || archetype == 2u) {
                float3 cabinInterior = float3(0.02, 0.025, 0.03);
                float3 R = reflect(curDir, normal);
                float3 envRefl = evalStudioEnvironment(R);
                float cosTheta = clamp(dot(-curDir, normal), 0.0, 1.0);
                float F = 0.04 + 0.96 * pow(1.0 - cosTheta, 5.0);
                directLight = lerp(cabinInterior, envRefl, F);
                throughput = float3(0.0);
                terminatePath = true;
            } else {
                if (pc.sceneType == SCENE_SHOWROOM_STUDIO) {
                    for (int i = 0; i < 3; ++i) {
                        float3 sDir, sCol;
                        getStudioLight(i, sDir, sCol);
                        float sNdotL = max(0.0, dot(normal, sDir));
                        if (sNdotL > 0.0) {
                            if (!traceShadowRay(topLevelAS, rq, hitPos, normal, sDir, 5000.0)) {
                                if (pc.isGltf != 0u) {
                                    directLight += evaluateCookTorranceGGX(albedo, roughness, metallic, normal, -curDir, sDir, sCol);
                                } else {
                                    directLight += (albedo / 3.14159) * sCol * sNdotL;
                                }
                            }
                        }
                    }
                } else {
                    float3 sunDir, sunColor;
                    getSceneSunParams(pc.sceneType, sunDir, sunColor);
                    float NdotL = max(0.0, dot(normal, sunDir));
                    if (NdotL > 0.0) {
                        if (!traceShadowRay(topLevelAS, rq, hitPos, normal, sunDir, (pc.isGltf != 0u) ? 5000.0 : 2000.0)) {
                            if (pc.isGltf != 0u) {
                                directLight = evaluateCookTorranceGGX(albedo, roughness, metallic, normal, -curDir, sunDir, sunColor);
                            } else {
                                directLight = (albedo / 3.14159) * sunColor * NdotL;
                            }
                        }
                    }
                }

                if (pc.sceneType == SCENE_INDOOR_ATRIUM) {
                    for (int li = 0; li < 3; ++li) {
                        float3 lPos, lCol;
                        getAtriumLantern(li, lPos, lCol);
                        float3 toL = lPos - hitPos;
                        float d2 = dot(toL, toL);
                        float d = sqrt(d2);
                        float3 L = toL / max(d, 0.001);
                        float lNdotL = max(0.0, dot(normal, L));
                        if (lNdotL > 0.0) {
                            if (!traceShadowRay(topLevelAS, rq, hitPos, normal, L, d - 0.1)) {
                                float atten = 1.0 / (1.0 + 0.000004 * d2);
                                directLight += (albedo / 3.14159) * lCol * atten * lNdotL;
                            }
                        }
                    }
                }

                if (metallic > 0.1 && u2 < metallic) {
                    float3 refl = reflect(curDir, normal);
                    float3 jitterVec = (float3(u1, u2, frac(u1 * 31.7)) * 2.0 - 1.0) * (roughness * 0.4);
                    bDir = normalize(refl + jitterVec);
                    if (dot(bDir, normal) <= 0.0) bDir = refl;
                    throughput = lerp(float3(0.04), albedo, metallic);
                } else {
                    bDir = nextDir;
                    throughput = (1.0 - metallic) * albedo;
                }
                bDir = quantizeOctDir(bDir);
            }

            float3 indirectRadiance = float3(0.0);
            uint maxBounces = pc.bounces;
            for (uint b = 1u; !terminatePath && b < maxBounces; ++b) {
                threadRays++;
                bool bHit = false;
                float b_t = 0.0;
                uint b_primId = 0u;
                float2 b_bary = float2(0.0);
                {
                    RayDesc bRay = { bOrigin, rayOffset, bDir, tMaxPrimary };
                    rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, bRay);
                    while (rq.Proceed()) {}
                    if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                        bHit = true;
                        b_t = rq.CommittedRayT();
                        b_primId = rq.CommittedPrimitiveIndex();
                        b_bary = rq.CommittedTriangleBarycentrics();
                    }
                }

                if (bHit) {
                    threadHits++;
                    float3 b_hitPos = bOrigin + bDir * b_t;
                    float3 b_normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, b_primId, b_hitPos, bDir, b_bary);

                    float3 b_albedo = float3(0.72);
                    if (pc.isGltf != 0u) {
                        float2 b_ubary = unpackUnorm2x16_spv(packUnorm2x16_spv(b_bary));
                        float2 b_uv = unpackGltfUV(vbuf, b_primId, b_ubary);
                        uint b_matId = triMatBuf[b_primId];
                        GltfMaterialGpu b_mat = matBuf[b_matId];
                        b_albedo = b_mat.baseColorFactor.rgb;
                        if (b_mat.baseColorTexIdx >= 0) {
                            b_albedo *= pow(sampleTexture(texHeaders, texPixels, uint(b_mat.baseColorTexIdx), b_uv).rgb, float3(2.2));
                        }
                    } else {
                        uint b_matId = computeMaterialId(pc.sceneType, b_primId, b_hitPos, b_normal);
                        b_albedo = getSceneArchetypeAlbedo(pc.sceneType, b_matId);
                    }

                    indirectRadiance += throughput * evalPathBounceDirectLight(topLevelAS, rq, pc.sceneType, pc.isGltf, b_hitPos, b_normal, b_albedo, true);

                    if (b + 1u < maxBounces) {
                        uint bRng = pixelIdx + pc.seed + b * 1337u;
                        if (b >= 2u && randFloat(bRng) < 0.4) break;

                        throughput *= b_albedo;
                        float b_u1 = randFloat(bRng);
                        float b_u2 = randFloat(bRng);
                        float3 nextBDir = alignToNormal(cosineSampleHemisphere(b_u1, b_u2), b_normal);
                        bDir = quantizeOctDir(nextBDir);
                        bOrigin = b_hitPos + b_normal * rayOffset;
                    }
                } else {
                    indirectRadiance += throughput * evalSky(pc.sceneType, bDir);
                    break;
                }
            }
            sampleRadiance = emissive + directLight + indirectRadiance;
        } else {
            sampleRadiance = evalSky(pc.sceneType, curDir);
        }

        if (pc.dumpRenders != 0u) {
            float expGain = (pc.sceneType == SCENE_INDOOR_ATRIUM) ? 1.6 : 1.0;
            if (totalSpp <= 1u) {
                fb[pixelIdx] = float4(tonemapACES(sampleRadiance * expGain), 1.0);
            } else if (sampleIdx == 0u) {
                fb[pixelIdx] = float4(sampleRadiance, 1.0);
            } else if (sampleIdx < totalSpp - 1u) {
                fb[pixelIdx] = fb[pixelIdx] + float4(sampleRadiance, 0.0);
            } else {
                float3 total = fb[pixelIdx].rgb + sampleRadiance;
                float3 avg = (total / float(totalSpp)) * expGain;
                fb[pixelIdx] = float4(tonemapACES(avg), 1.0);
            }
        } else {
            threadLum = luminanceToUint(sampleRadiance);
        }
    } else if (mode == 2u) {
        // Incoherent Secondary GI
        threadRays++;
        bool hit1 = false;
        float t1 = 0.0;
        uint prim1 = 0u;
        float2 bary1 = float2(0.0);
        {
            RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                hit1 = true;
                t1 = rq.CommittedRayT();
                prim1 = rq.CommittedPrimitiveIndex();
                bary1 = rq.CommittedTriangleBarycentrics();
            }
        }

        if (hit1) {
            threadHits++;
            float3 hitPos = rayOrigin + rayDir * t1;
            float3 normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, prim1, hitPos, rayDir, bary1);
            float tMaxSec = (pc.isGltf != 0u) ? 5000.0 : ((pc.sceneType == SCENE_OUTDOOR_TERRAIN) ? 500.0 : 20.0);
            float rayOffset = (pc.isGltf != 0u) ? 0.05 : 0.005;

            float u1 = randFloat(rngState);
            float u2 = randFloat(rngState);
            float3 incohDir = alignToNormal(cosineSampleHemisphere(u1, u2), normal);
            uint curPackedDir = packSnorm2x16_spv(octEncode(incohDir));
            float3 curDir = octDecode(unpackSnorm2x16_spv(curPackedDir));
            float3 curOrigin = hitPos + normal * rayOffset;

            for (uint b = 1u; b <= 4u; ++b) {
                threadRays++;
                RayDesc secRay = { curOrigin, rayOffset, curDir, tMaxSec };
                rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, secRay);
                while (rq.Proceed()) {}
                if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
                    break;
                }
                threadHits++;
                if (b < 4u) {
                    uint h = giRayHash(curOrigin, curPackedDir, b);
                    if (b >= 2u && randFloat(h) < 0.4) {
                        break;
                    }
                    float b_t = rq.CommittedRayT();
                    uint b_primId = rq.CommittedPrimitiveIndex();
                    float2 b_bary = rq.CommittedTriangleBarycentrics();
                    float3 b_hitPos = curOrigin + curDir * b_t;
                    float3 b_normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, b_primId, b_hitPos, curDir, b_bary);
                    float bu1 = randFloat(h);
                    float bu2 = randFloat(h);
                    float3 nextDir = alignToNormal(cosineSampleHemisphere(bu1, bu2), b_normal);
                    curPackedDir = packSnorm2x16_spv(octEncode(nextDir));
                    curDir = octDecode(unpackSnorm2x16_spv(curPackedDir));
                    curOrigin = b_hitPos + b_normal * rayOffset;
                }
            }
        }
    } else if (mode == 3u) {
        // Pure BVH Traversal Only
        threadRays++;
        {
            RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                threadHits++;
            }
        }
    } else if (mode == 4u) {
        // Pure Material Shading (Megakernel Divergent, No BVH)
        threadRays++;
        threadHits++;
        float3 pixelColor;
        uint archId = pixelIdx % 8u;
        float3 dummyPos = float3(float(px_u) * 0.01, float(py_u) * 0.01, 1.5);
        float3 dummyNorm = ((pixelIdx % 2u) == 0u) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        if (pc.isGltf != 0u) {
            uint matCount, matStride;
            matBuf.GetDimensions(matCount, matStride);
            matCount = max(matCount, 1u);
            GltfMaterialGpu mat = matBuf[archId % matCount];
            mat.archetype = archId;
            float4 dummyTan = float4(1.0, 0.0, 0.0, 1.0);
            float2 dummyUv = float2(float(px_u) / float(pc.width), float(py_u) / float(pc.height));
            pixelColor = evaluateGltfPbrShading(texHeaders, texPixels, 0xFFFFFFFFu, mat, dummyPos, dummyNorm, dummyNorm, dummyTan, dummyUv, rayDir, pc.sceneType, false, false, float3(1.0, 1.0, -1.0));
        } else {
            pixelColor = evalProceduralMaterialBSDF(pc.sceneType, archId, 0xFFFFFFFFu, dummyNorm, rayDir, dummyPos, false, false, 1.0, uint3(0xFFFFFFFFu, 0u, 0u));
        }
        if (pc.dumpRenders != 0u) {
            fb[pixelIdx] = float4(tonemapACES(pixelColor), 1.0);
        } else {
            threadLum = luminanceToUint(pixelColor);
        }
    } else if (mode == 5u) {
        // Directional Shadow Ray Traversal
        threadRays++;
        bool hitFound = false;
        float t = 0.0;
        uint primId = 0u;
        float2 rawBary = float2(0.0);
        {
            RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                hitFound = true;
                t = rq.CommittedRayT();
                primId = rq.CommittedPrimitiveIndex();
                rawBary = rq.CommittedTriangleBarycentrics();
            }
        }

        if (hitFound) {
            threadHits++;
            float3 hitPos = rayOrigin + rayDir * t;
            float3 geomNormal = (pc.isGltf != 0u)
                ? computeGltfGeomNormal(vbuf, primId, rayDir)
                : computeSmoothNormal(vbuf, pc.sceneType, 0u, primId, hitPos, rayDir, rawBary);
            geomNormal = quantizeOctDir(geomNormal);

            uint lightIdx = (px_u + py_u * 2u) % 3u;
            float3 lightDir;
            float maxDist;
            getShadowLightParams(pc.sceneType, lightIdx, lightDir, maxDist);

            float shadowVal = 0.95;
            if (dot(geomNormal, lightDir) > 0.0) {
                threadRays += 4u;
                uint shRng = pixelIdx + pc.seed;
                float vis = traceAreaShadow(topLevelAS, rq, hitPos, geomNormal, lightDir, maxDist, 0.045, shRng, 4);
                threadHits += uint(round((1.0 - vis) * 4.0));
                shadowVal = lerp(0.08, 0.95, vis);
            }

            if (pc.dumpRenders != 0u) {
                fb[pixelIdx] = float4(float3(shadowVal), 1.0);
            } else {
                threadLum = uint(clamp(shadowVal, 0.0, 1.0) * 255.0 + 0.5);
            }
        } else {
            if (pc.dumpRenders != 0u) {
                fb[pixelIdx] = float4(0.02, 0.03, 0.05, 1.0);
            }
        }
    } else if (mode == 6u) {
        // BVH Traversal Heatmap
        threadRays++;
        uint steps = 0u;
        bool hit = false;
        float hitT = 0.0;
        uint primId = 0u;
        float2 rawBary = float2(0.0);
        {
            RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_NON_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {
                steps++;
                if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                    rq.CommitNonOpaqueTriangleHit();
                }
            }
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                hit = true;
                hitT = rq.CommittedRayT();
                primId = rq.CommittedPrimitiveIndex();
                rawBary = rq.CommittedTriangleBarycentrics();
            }
        }
        if (hit) {
            threadHits++;
        }
        if (pc.dumpRenders != 0u) {
            float3 col = float3(0.015, 0.02, 0.035);
            if (hit) {
                float3 hitPos = rayOrigin + rayDir * hitT;
                float3 normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, primId, hitPos, rayDir, rawBary);
                float tSteps = clamp(log2(max(1.0, float(steps))) / 4.0, 0.0, 1.0);
                float3 heatCol = turboColormap(tSteps);
                float3 shadeDir = normalize(float3(0.35, 0.55, 0.75));
                float diff = clamp(dot(normal, shadeDir) * 0.25 + 0.75, 0.45, 1.0);
                col = heatCol * diff;
            }
            fb[pixelIdx] = float4(col, 1.0);
        } else {
            threadLum = steps;
        }
    } else if (mode == 7u) {
        // Primary Surface Normals (G-Buffer)
        threadRays++;
        bool hitFound = false;
        float t = 0.0;
        uint primId = 0u;
        float2 rawBary = float2(0.0);
        {
            RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                hitFound = true;
                t = rq.CommittedRayT();
                primId = rq.CommittedPrimitiveIndex();
                rawBary = rq.CommittedTriangleBarycentrics();
            }
        }
        if (hitFound) {
            threadHits++;
            float3 hitPos = rayOrigin + rayDir * t;
            float2 bary = (pc.isGltf != 0u) ? unpackUnorm2x16_spv(packUnorm2x16_spv(rawBary)) : rawBary;
            float3 normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, primId, hitPos, rayDir, bary);
            float3 nCol = clamp(normal * 0.5 + 0.5, 0.0, 1.0);
            if (pc.dumpRenders != 0u) {
                fb[pixelIdx] = float4(nCol, 1.0);
            } else {
                threadLum = luminanceToUint(nCol);
            }
        } else {
            if (pc.dumpRenders != 0u) {
                fb[pixelIdx] = float4(0.04, 0.05, 0.08, 1.0);
            }
        }
    } else if (mode == 8u) {
        // Isolated Directional Shadow Mask
        threadRays++;
        bool hitFound = false;
        float t = 0.0;
        uint primId = 0u;
        float2 rawBary = float2(0.0);
        {
            RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                hitFound = true;
                t = rq.CommittedRayT();
                primId = rq.CommittedPrimitiveIndex();
                rawBary = rq.CommittedTriangleBarycentrics();
            }
        }
        if (hitFound) {
            threadHits++;
            float3 hitPos = rayOrigin + rayDir * t;
            float3 normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, primId, hitPos, rayDir, rawBary);
            float3 sunDir, sunColor;
            getSceneSunParams(pc.sceneType, sunDir, sunColor);
            bool shadowed = true;
            if (dot(normal, sunDir) > 0.0) {
                shadowed = traceShadowRay(topLevelAS, rq, hitPos, normal, sunDir, (pc.isGltf != 0u) ? 5000.0 : 2000.0);
            }
            if (pc.dumpRenders != 0u) {
                fb[pixelIdx] = float4(float3(shadowed ? 0.08 : 0.98), 1.0);
            } else {
                threadLum = shadowed ? 20u : 250u;
            }
        } else {
            if (pc.dumpRenders != 0u) {
                fb[pixelIdx] = float4(0.98, 0.98, 1.0, 1.0);
            }
        }
    } else if (mode == 9u) {
        // Isolated RTAO Contact Occlusion
        threadRays++;
        bool hitFound = false;
        float t = 0.0;
        uint primId = 0u;
        float2 rawBary = float2(0.0);
        {
            RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                hitFound = true;
                t = rq.CommittedRayT();
                primId = rq.CommittedPrimitiveIndex();
                rawBary = rq.CommittedTriangleBarycentrics();
            }
        }
        if (hitFound) {
            threadHits++;
            float3 hitPos = rayOrigin + rayDir * t;
            float3 normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, primId, hitPos, rayDir, rawBary);
            uint aoRng = pixelIdx + pc.seed;
            float rtao = computeRTAO(topLevelAS, rq, hitPos, normal, aoRng, 4, (pc.isGltf != 0u) ? 15.0 : 35.0);
            if (pc.dumpRenders != 0u) {
                fb[pixelIdx] = float4(float3(rtao), 1.0);
            } else {
                threadLum = uint(clamp(rtao, 0.0, 1.0) * 256.0);
            }
        } else {
            if (pc.dumpRenders != 0u) {
                fb[pixelIdx] = float4(1.0, 1.0, 1.0, 1.0);
            }
        }
    } else if (mode == 10u) {
        // Isolated Secondary Indirect Diffuse GI Bounce
        threadRays++;
        bool hitFound = false;
        float t = 0.0;
        uint primId = 0u;
        float2 rawBary = float2(0.0);
        {
            RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                hitFound = true;
                t = rq.CommittedRayT();
                primId = rq.CommittedPrimitiveIndex();
                rawBary = rq.CommittedTriangleBarycentrics();
            }
        }
        if (hitFound) {
            threadHits++;
            float3 hitPos = rayOrigin + rayDir * t;
            float3 normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, primId, hitPos, rayDir, rawBary);

            float3 albedo = float3(0.72);
            if (pc.isGltf != 0u) {
                float2 bary = unpackUnorm2x16_spv(packUnorm2x16_spv(rawBary));
                float2 uv = unpackGltfUV(vbuf, primId, bary);
                uint matId = triMatBuf[primId];
                GltfMaterialGpu mat = matBuf[matId];
                albedo = mat.baseColorFactor.rgb;
                if (mat.baseColorTexIdx >= 0) {
                    albedo *= pow(sampleTexture(texHeaders, texPixels, uint(mat.baseColorTexIdx), uv).rgb, float3(2.2));
                }
            } else {
                uint matId = computeMaterialId(pc.sceneType, primId, hitPos, normal);
                albedo = getSceneArchetypeAlbedo(pc.sceneType, matId);
            }

            float3 sunDir, sunColor;
            getSceneSunParams(pc.sceneType, sunDir, sunColor);

            float3 throughput = albedo;
            float3 accumIndirect = float3(0.0);
            uint giRng = pixelIdx + pc.seed;
            const uint kBounceSamples = 4u;
            float rayOffset = (pc.isGltf != 0u) ? 0.05 : 0.005;

            for (uint s = 0u; s < kBounceSamples; ++s) {
                float u1 = (float(s) + randFloat(giRng)) / float(kBounceSamples);
                float u2 = randFloat(giRng);
                float3 bDir = alignToNormal(cosineSampleHemisphere(u1, u2), normal);
                float3 bOrigin = hitPos + normal * rayOffset;

                bool bHit = false;
                float b_t = 0.0;
                uint b_primId = 0u;
                float2 b_bary = float2(0.0);
                {
                    RayDesc bRay = { bOrigin, rayOffset, bDir, tMaxPrimary };
                    rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, bRay);
                    while (rq.Proceed()) {}
                    if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                        bHit = true;
                        b_t = rq.CommittedRayT();
                        b_primId = rq.CommittedPrimitiveIndex();
                        b_bary = rq.CommittedTriangleBarycentrics();
                    }
                }

                if (bHit) {
                    float3 b_hitPos = bOrigin + bDir * b_t;
                    float3 b_normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, b_primId, b_hitPos, bDir, b_bary);
                    float3 b_albedo = float3(0.72);
                    if (pc.isGltf != 0u) {
                        float2 b_ubary = unpackUnorm2x16_spv(packUnorm2x16_spv(b_bary));
                        float2 b_uv = unpackGltfUV(vbuf, b_primId, b_ubary);
                        uint b_matId = triMatBuf[b_primId];
                        GltfMaterialGpu b_mat = matBuf[b_matId];
                        b_albedo = b_mat.baseColorFactor.rgb;
                        if (b_mat.baseColorTexIdx >= 0) {
                            b_albedo *= pow(sampleTexture(texHeaders, texPixels, uint(b_mat.baseColorTexIdx), b_uv).rgb, float3(2.2));
                        }
                    } else {
                        uint b_matId = computeMaterialId(pc.sceneType, b_primId, b_hitPos, b_normal);
                        b_albedo = getSceneArchetypeAlbedo(pc.sceneType, b_matId);
                    }

                    float3 b_directLight = float3(0.0);
                    float b_NdotL = max(0.0, dot(b_normal, sunDir));
                    if (b_NdotL > 0.0) {
                        if (!traceShadowRay(topLevelAS, rq, b_hitPos, b_normal, sunDir, (pc.isGltf != 0u) ? 5000.0 : 2000.0)) {
                            b_directLight += (b_albedo / 3.14159) * sunColor * b_NdotL;
                        }
                    }

                    if (pc.sceneType == SCENE_INDOOR_ATRIUM) {
                        for (int li = 0; li < 3; ++li) {
                            float3 lPos, lCol;
                            getAtriumLantern(li, lPos, lCol);
                            float3 b_toL = lPos - b_hitPos;
                            float bd2 = dot(b_toL, b_toL);
                            float bd = sqrt(bd2);
                            float3 b_L = b_toL / max(bd, 0.001);
                            float bl_NdotL = max(0.0, dot(b_normal, b_L));
                            if (bl_NdotL > 0.0) {
                                if (!traceShadowRay(topLevelAS, rq, b_hitPos, b_normal, b_L, bd - 0.1)) {
                                    float atten = 1.0 / (1.0 + 0.000004 * bd2);
                                    b_directLight += (b_albedo / 3.14159) * lCol * atten * bl_NdotL;
                                }
                            }
                        }
                    } else if (pc.sceneType == SCENE_SHOWROOM_STUDIO) {
                        for (int l = 1; l < 3; ++l) {
                            float3 slDir, slCol;
                            float slInt;
                            getSceneLight(l, slDir, slCol, slInt);
                            float3 fillDir = normalize(slDir);
                            float f_NdotL = max(0.0, dot(b_normal, fillDir));
                            if (f_NdotL > 0.0) {
                                if (!traceShadowRay(topLevelAS, rq, b_hitPos, b_normal, fillDir, 50.0)) {
                                    b_directLight += (b_albedo / 3.14159) * slCol * slInt * f_NdotL;
                                }
                            }
                        }
                    }

                    accumIndirect += throughput * b_directLight;
                }
            }

            float3 indirectRadiance = accumIndirect / float(kBounceSamples);
            if (pc.dumpRenders != 0u) {
                fb[pixelIdx] = float4(tonemapACES(indirectRadiance * 3.5), 1.0);
            } else {
                threadLum = luminanceToUint(indirectRadiance);
            }
        } else {
            if (pc.dumpRenders != 0u) {
                fb[pixelIdx] = float4(0.0, 0.0, 0.0, 1.0);
            }
        }
    } else if (mode == 11u) {
        // Multi-Light Evaluation (Monolithic Megakernel Baseline)
        threadRays++;
        bool hitFound = false;
        float t = 0.0;
        uint primId = 0u;
        float2 rawBary = float2(0.0);
        {
            RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                hitFound = true;
                t = rq.CommittedRayT();
                primId = rq.CommittedPrimitiveIndex();
                rawBary = rq.CommittedTriangleBarycentrics();
            }
        }

        float3 totalRadiance = float3(0.0);
        if (hitFound) {
            threadHits++;
            float3 hitPos = rayOrigin + rayDir * t;
            float3 viewDir = -rayDir;
            float3 smoothNormal;
            float3 albedo = float3(0.72);
            float roughness = 0.5;
            float metallic = 0.0;

            if (pc.isGltf != 0u) {
                float2 bary = unpackUnorm2x16_spv(packUnorm2x16_spv(rawBary));
                float3 interpPos, geomNormal;
                float4 dummyTan;
                float2 uv;
                unpackGltfHit(vbuf, primId, bary, false, interpPos, geomNormal, smoothNormal, dummyTan, uv);

                uint matId = triMatBuf[primId];
                GltfMaterialGpu mat = matBuf[matId];
                albedo = mat.baseColorFactor.rgb;
                roughness = mat.roughnessFactor;
                metallic = mat.metallicFactor;
                if (mat.baseColorTexIdx >= 0) {
                    albedo *= pow(sampleTexture(texHeaders, texPixels, uint(mat.baseColorTexIdx), uv).rgb, float3(2.2));
                }
            } else {
                smoothNormal = computeSmoothNormal(vbuf, pc.sceneType, 0u, primId, hitPos, rayDir, rawBary);
                if (pc.sceneType == SCENE_OUTDOOR_TERRAIN) {
                    getProceduralTerrainMaterial(primId, hitPos, smoothNormal, albedo, roughness, metallic);
                }
            }

            smoothNormal = quantizeOctDir(smoothNormal);
            metallic = float(uint(clamp(metallic, 0.0, 1.0) * 255.0 + 0.5)) * (1.0 / 255.0);
            float2 albRG = unpackHalf2x16_spv(packHalf2x16_spv(albedo.xy));
            float2 albBR = unpackHalf2x16_spv(packHalf2x16_spv(float2(albedo.z, roughness)));
            albedo = float3(albRG, albBR.x);
            roughness = albBR.y;

            uint totalLights = pc.spp;
            if (totalLights <= 1u) {
                float3 L, C;
                getMultiLight(0u, 1u, pc.sceneType, L, C);
                totalRadiance += evalCookTorranceDirect(smoothNormal, viewDir, L, C, albedo, roughness, metallic);
            } else {
                for (uint b = 0u; b < 8u; ++b) {
                    uint startLight = b * 16u;
                    for (uint li = 0u; li < 16u; ++li) {
                        float3 L, C;
                        getMultiLight(startLight + li, 128u, pc.sceneType, L, C);
                        totalRadiance += evalCookTorranceDirect(smoothNormal, viewDir, L, C, albedo, roughness, metallic);
                    }
                }
            }
        } else {
            totalRadiance = evalSky(pc.sceneType, rayDir) * 0.15;
        }

        if (pc.dumpRenders != 0u) {
            fb[pixelIdx] = float4(tonemapACES(totalRadiance), 1.0);
        } else {
            threadLum = luminanceToUint(totalRadiance);
        }
    } else if (mode == 12u) {
        // Pipeline Breakdown - Isolated Direct Hybrid PBR Shading
        threadRays++;
        bool hitFound = false;
        float t = 0.0;
        uint primId = 0u;
        float2 rawBary = float2(0.0);
        {
            RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                hitFound = true;
                t = rq.CommittedRayT();
                primId = rq.CommittedPrimitiveIndex();
                rawBary = rq.CommittedTriangleBarycentrics();
            }
        }

        float3 pixelColor = float3(0.0);
        if (hitFound) {
            threadHits++;
            float3 hitPos = rayOrigin + rayDir * t;
            uint rngP = pixelIdx;
            if (pc.isGltf != 0u) {
                float2 bary = unpackUnorm2x16_spv(packUnorm2x16_spv(rawBary));
                float3 interpHitPos, geomNormal, smoothNormal;
                float4 tangent;
                float2 uv;
                unpackGltfHit(vbuf, primId, bary, true, interpHitPos, geomNormal, smoothNormal, tangent, uv);
                uint matId = triMatBuf[primId];
                GltfMaterialGpu mat = matBuf[matId];
                pixelColor = evaluateGltfPbrWithRayQuery(topLevelAS, rq, texHeaders, texPixels, 0xFFFFFFFFu, mat, interpHitPos, geomNormal, smoothNormal, tangent, uv, rayDir, pc.sceneType, true, true, rngP);
            } else {
                float3 normal = computeSmoothNormal(vbuf, pc.sceneType, 0u, primId, hitPos, rayDir, rawBary);
                uint matId = computeMaterialId(pc.sceneType, primId, hitPos, normal);
                if (pc.sceneType == SCENE_OUTDOOR_TERRAIN) {
                    normal = quantizeOctDir(normal);
                }
                pixelColor = evalProceduralMaterialBSDFWithRayQuery(topLevelAS, rq, pc.sceneType, matId, 0xFFFFFFFFu, normal, rayDir, hitPos, true, true, rngP);
            }
        } else {
            pixelColor = evalSky(pc.sceneType, rayDir) * 0.15;
        }

        if (pc.dumpRenders != 0u) {
            fb[pixelIdx] = float4(tonemapACES(pixelColor), 1.0);
        } else {
            threadLum = luminanceToUint(pixelColor);
        }
    } else {
        // Mode 0: Primary Ray Tracing + Divergent Material Shading
        threadRays++;
        bool hitFound = false;
        float t = 0.0;
        uint primId = 0u;
        float2 rawBary = float2(0.0);
        {
            RayDesc ray = { rayOrigin, tMinPrimary, rayDir, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                hitFound = true;
                t = rq.CommittedRayT();
                primId = rq.CommittedPrimitiveIndex();
                rawBary = rq.CommittedTriangleBarycentrics();
            }
        }

        float3 pixelColor = float3(0.0);
        if (hitFound) {
            threadHits++;
            float3 hitPos = rayOrigin + rayDir * t;
            uint rngP = pixelIdx;
            if (pc.isGltf != 0u) {
                uint matId = triMatBuf[primId];
                GltfMaterialGpu mat = matBuf[matId];
                float2 bary = unpackUnorm2x16_spv(packUnorm2x16_spv(rawBary));
                float3 interpHitPos, geomNormal, smoothNormal;
                float4 tangent;
                float2 uv;
                unpackGltfHit(vbuf, primId, bary, mat.normalTexIdx >= 0, interpHitPos, geomNormal, smoothNormal, tangent, uv);
                pixelColor = evaluateGltfPbrWithRayQuery(topLevelAS, rq, texHeaders, texPixels, 0xFFFFFFFFu, mat, interpHitPos, geomNormal, smoothNormal, tangent, uv, rayDir, pc.sceneType, true, false, rngP);
            } else {
                float3 normal = computeSmoothNormal(vbuf, pc.sceneType, 0u, primId, hitPos, rayDir, rawBary);
                uint matId = computeMaterialId(pc.sceneType, primId, hitPos, normal);
                if (pc.sceneType == SCENE_OUTDOOR_TERRAIN) {
                    normal = quantizeOctDir(normal);
                }
                pixelColor = evalProceduralMaterialBSDFWithRayQuery(topLevelAS, rq, pc.sceneType, matId, 0xFFFFFFFFu, normal, rayDir, hitPos, true, false, rngP);
            }
        } else {
            pixelColor = evalSky(pc.sceneType, rayDir);
        }

        if (pc.dumpRenders != 0u) {
            fb[pixelIdx] = float4(tonemapACES(pixelColor), 1.0);
        } else {
            threadLum = luminanceToUint(pixelColor);
        }
    }

    addWaveCounters(results, threadRays, threadHits, threadLum);
}
