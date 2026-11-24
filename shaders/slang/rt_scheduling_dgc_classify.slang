import rt_types;
import rt_math;
import rt_scene;
import rt_pbr;

[[vk::constant_id(0)]] const uint CLASSIFY_MODE = 0xFFFFFFFFu;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] RWStructuredBuffer<Results> results;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint4> worklist;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> fb;
[[vk::binding(5, 0)]] StructuredBuffer<float> vbuf;
[[vk::binding(6, 0)]] StructuredBuffer<GltfMaterialGpu> matBuf;
[[vk::binding(7, 0)]] StructuredBuffer<uint> triMatBuf;
[[vk::binding(8, 0)]] StructuredBuffer<TextureHeaderGpu> texHeaders;
[[vk::binding(9, 0)]] StructuredBuffer<uint> texPixels;

struct PushConstants {
    uint rayCount;
    uint mode; // 0: MatDiv, 1: PT, 2: IncohGI, 3: Compaction, 5: FullStack/Shadows, 6: DirShadow, 7: AlphaCutout, 8: MultiLight, 9: PrimaryGBuffer
    uint bounce;
    uint seed;
    uint dumpRenders;
    uint width;
    uint height;
    uint queueCapacity;
    uint spatialPattern;
    uint sceneType;
    uint isGltf;
    uint spp;
    uint bounces;
};

[[vk::push_constant]] PushConstants pc;

[numthreads(8, 4, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint gidx : SV_GroupIndex, uint3 wgid : SV_GroupID) {
    const uint mode = (CLASSIFY_MODE != 0xFFFFFFFFu) ? CLASSIFY_MODE : pc.mode;
    uint3 nwg = WorkgroupCount();
    uint idx = (nwg.y > 1u) ? (gid.y * pc.width + gid.x) : (wgid.x * 32u + gidx);
    if (mode == 1u && idx == 0u) {
        wlStoreSeed(worklist, pc.seed);
        wlStoreMeta(worklist, (pc.dumpRenders & 0xFFu) | ((pc.spp & 0xFFu) << 8u) | ((pc.bounce & 0xFFu) << 16u));
    }

    uint tileIdx = (nwg.y > 1u) ? (idx >> 5u) : wgid.x;
    uint inTile = (nwg.y > 1u) ? (idx & 31u) : gidx;
    uint px_u, py_u;
    computePixelCoords(idx, tileIdx, inTile, (CLASSIFY_MODE != 0xFFFFFFFFu) ? 2u : pc.spatialPattern, pc.width, pc.height, pc.isGltf, px_u, py_u);
    uint pixelIdx = py_u * pc.width + px_u;

    uint threadRays = 0u;
    uint threadHits = 0u;
    uint threadLum = 0u;
    uint assignedQueue = 0xFFFFFFFFu;
    float4 payload0 = float4(0.0);
    float4 payload1 = float4(0.0);

    if (idx < pc.rayCount) {
        threadRays = 1u;
        uint rngState = pixelIdx + pc.seed;

        Camera cam = getCamera(pc.sceneType, pc.isGltf);
        float2 jitter = float2(0.0, 0.0);
        if (mode == 1u && pc.spp > 1u) {
            jitter.x = randFloat(rngState) - 0.5;
            jitter.y = randFloat(rngState) - 0.5;
        }
        float3 origin = cam.pos;
        float3 direction = makePrimaryRayDir(cam, px_u, py_u, pc.width, pc.height, jitter);
        float tMinPrimary, tMaxPrimary;
        getPrimaryRayRange(pc.sceneType, pc.isGltf, tMinPrimary, tMaxPrimary);

        RayQuery<RAY_FLAG_NONE> rq;
        bool primaryHit = false;
        float t = 0.0;
        uint primId = 0u;
        float2 rawBary = float2(0.0);

        if (mode != 7u) {
            RayDesc ray = { origin, tMinPrimary, direction, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                primaryHit = true;
                t = rq.CommittedRayT();
                primId = rq.CommittedPrimitiveIndex();
                rawBary = (mode == 2u && pc.isGltf == 0u) ? float2(0.0) : rq.CommittedTriangleBarycentrics();
            }
        } else {
            RayDesc ray = { origin, tMinPrimary, direction, tMaxPrimary };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFFu, ray);
            while (rq.Proceed()) {
                if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                    uint candTri = rq.CandidatePrimitiveIndex();
                    float2 candBary = rq.CandidateTriangleBarycentrics();
                    if (pc.isGltf != 0u) {
                        uint candMatId = triMatBuf[candTri];
                        GltfMaterialGpu candMat = matBuf[candMatId];
                        float2 cuv = unpackGltfUV(vbuf, candTri, candBary);
                        if (candMat.alphaMode == 1u) {
                            float calpha = candMat.baseColorFactor.a;
                            if (candMat.baseColorTexIdx >= 0) {
                                calpha *= sampleTexture(texHeaders, texPixels, uint(candMat.baseColorTexIdx), cuv).a;
                            } else {
                                float cw = 1.0 - candBary.x - candBary.y;
                                float frondPattern = frac(sin(dot(cuv * 24.0 + float2(float(candTri & 63u) * 0.13), float2(12.9898, 78.233))) * 43758.5453);
                                float edgeDist = min(cw, min(candBary.x, candBary.y)) * 3.0;
                                calpha = lerp(frondPattern, edgeDist, 0.35);
                            }
                            if (calpha < candMat.alphaCutoff) {
                                continue;
                            }
                        } else if (pc.sceneType == SCENE_SHOWROOM_STUDIO) {
                            float perf = frac(sin(dot(candBary * 16.0 + float2(float(candTri & 127u) * 0.17), float2(12.9898, 78.233))) * 43758.5453);
                            if ((candTri & 1u) == 0u && perf < 0.45) {
                                continue;
                            }
                        }
                    } else {
                        float leafAlpha = frac(sin(dot(candBary * 18.0 + float2(float(candTri & 127u) * 0.19), float2(12.9898, 78.233))) * 43758.5453);
                        if ((candTri >= PRIM_WATER_END || (candTri & 1u) == 0u) && leafAlpha < 0.48) {
                            continue;
                        }
                    }
                    rq.CommitNonOpaqueTriangleHit();
                }
            }
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                primaryHit = true;
                t = rq.CommittedRayT();
                primId = rq.CommittedPrimitiveIndex();
                rawBary = rq.CommittedTriangleBarycentrics();
            }
        }

        if (primaryHit) {
            threadHits = 1u;
            float3 hitPos = origin + direction * t;
            if (WaveActiveAllEqual(primId)) {
                primId = WaveReadLaneFirst(primId);
            }

            if (mode == 3u) {
                if (pc.isGltf != 0u) {
                    uint matId = triMatBuf[primId];
                    assignedQueue = matBuf[matId].archetype % 8u;
                } else {
                    assignedQueue = primId % 8u;
                }
                payload0 = float4(t, asfloat(pixelIdx), asfloat(primId), asfloat(packUnorm2x16_spv(rawBary)));
            } else if (mode == 5u) {
                uint q = (pc.spp == 3u) ? ((wgid.x >> 6u) % 3u) : 0u;
                float3 geomNormal = (pc.isGltf != 0u)
                    ? computeGltfGeomNormal(vbuf, primId, direction)
                    : computeSmoothNormal(vbuf, pc.sceneType, 0u, primId, hitPos, direction, rawBary);
                float3 lightDir;
                float unusedDist;
                getShadowLightParams(pc.sceneType, q, lightDir, unusedDist);

                if (dot(geomNormal, lightDir) > 0.0) {
                    threadRays += 1u;
                    assignedQueue = q;
                    payload0 = float4(hitPos + geomNormal * 0.12, asfloat(pixelIdx));
                } else {
                    assignedQueue = 0xFFFFFFFFu;
                    if (pc.dumpRenders != 0u) {
                        fb[pixelIdx] = float4(float3(0.95), 1.0);
                    } else {
                        threadLum = 243u;
                    }
                }
            } else if (mode == 8u) {
                float3 smoothNormal;
                float2 bary = float2(0.0);
                if (pc.isGltf != 0u) {
                    bary = unpackUnorm2x16_spv(packUnorm2x16_spv(rawBary));
                    // Preserve exact classify mode 8 unflipped normal sign when dot(n, dir) isn't checked
                    uint base = primId * 36u;
                    float3 n0 = float3(vbuf[base + 3u], vbuf[base + 4u], vbuf[base + 5u]);
                    float3 n1 = float3(vbuf[base + 15u], vbuf[base + 16u], vbuf[base + 17u]);
                    float3 n2 = float3(vbuf[base + 27u], vbuf[base + 28u], vbuf[base + 29u]);
                    float w = 1.0 - bary.x - bary.y;
                    float3 sn = w * n0 + bary.x * n1 + bary.y * n2;
                    float snLen2 = dot(sn, sn);
                    if (snLen2 > 1e-8) {
                        smoothNormal = sn * rsqrt(snLen2);
                    } else {
                        float3 p0 = float3(vbuf[base + 0u], vbuf[base + 1u], vbuf[base + 2u]);
                        float3 p1 = float3(vbuf[base + 12u], vbuf[base + 13u], vbuf[base + 14u]);
                        float3 p2 = float3(vbuf[base + 24u], vbuf[base + 25u], vbuf[base + 26u]);
                        smoothNormal = normalize(cross(p1 - p0, p2 - p0));
                    }
                } else {
                    smoothNormal = computeSmoothNormal(vbuf, pc.sceneType, 0u, primId, hitPos, direction, rawBary);
                }

                uint packedNorm = packSnorm2x16_spv(octEncode(smoothNormal));
                float3 qNorm = octDecode(unpackSnorm2x16_spv(packedNorm));

                bool needMat = true;
                if (pc.spp <= 1u) {
                    float3 L0, C0;
                    getMultiLight(0u, 1u, pc.sceneType, L0, C0);
                    if (dot(qNorm, L0) <= 0.0) {
                        needMat = false;
                        assignedQueue = 0xFFFFFFFFu;
                        if (pc.dumpRenders != 0u) {
                            fb[pixelIdx] = float4(0.0, 0.0, 0.0, 1.0);
                        }
                    }
                }

                if (needMat) {
                    float3 albedo = float3(0.72);
                    float roughness = 0.5;
                    float metallic = 0.0;
                    if (pc.isGltf != 0u) {
                        uint matId = triMatBuf[primId];
                        if (WaveActiveAllEqual(matId)) {
                            matId = WaveReadLaneFirst(matId);
                        }
                        GltfMaterialGpu mat = matBuf[matId];
                        albedo = mat.baseColorFactor.rgb;
                        roughness = mat.roughnessFactor;
                        metallic = mat.metallicFactor;
                        if (mat.baseColorTexIdx >= 0) {
                            float2 uv = unpackGltfUV(vbuf, primId, bary);
                            albedo *= pow(sampleTexture(texHeaders, texPixels, uint(mat.baseColorTexIdx), uv).rgb, float3(2.2));
                        }
                    } else if (pc.sceneType == SCENE_OUTDOOR_TERRAIN) {
                        getProceduralTerrainMaterial(primId, hitPos, smoothNormal, albedo, roughness, metallic);
                    }

                    uint packedPixMet = (pixelIdx & 0x00FFFFFFu) | (uint(clamp(metallic, 0.0, 1.0) * 255.0 + 0.5) << 24u);
                    uint packedAlbRG = packHalf2x16_spv(albedo.xy);
                    uint packedAlbBR = packHalf2x16_spv(float2(albedo.z, roughness));
                    uint octQ = (qNorm.x >= 0.0 ? 1u : 0u)
                              | (qNorm.y >= 0.0 ? 2u : 0u)
                              | (qNorm.z >= 0.0 ? 4u : 0u);
                    assignedQueue = (pc.spp > 1u) ? octQ : 0u;
                    payload0 = asfloat(uint4(packedNorm, packedPixMet, packedAlbRG, packedAlbBR));
                }
            } else if (mode == 6u || mode == 7u) {
                assignedQueue = 0u;
                float3 normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, primId, hitPos, direction, rawBary);
                float2 bary = unpackUnorm2x16_spv(packUnorm2x16_spv(rawBary));
                float2 hitUv = (pc.isGltf != 0u) ? unpackGltfUV(vbuf, primId, bary) : float2(0.0);

                payload0 = float4(hitPos, asfloat(packSnorm2x16_spv(octEncode(normal))));
                payload1 = float4(asfloat(packHalf2x16_spv(hitUv)), asfloat(packUnorm2x16_spv(rawBary)), asfloat(primId), asfloat(pixelIdx));

                if (mode == 7u) {
                    float3 albedo = float3(0.72);
                    if (pc.isGltf != 0u) {
                        uint matId = triMatBuf[primId];
                        GltfMaterialGpu mat = matBuf[matId];
                        albedo = mat.baseColorFactor.rgb;
                        if (mat.baseColorTexIdx >= 0) {
                            albedo *= pow(sampleTexture(texHeaders, texPixels, uint(mat.baseColorTexIdx), hitUv).rgb, float3(2.2));
                        }
                    } else {
                        uint matId = computeMaterialId(pc.sceneType, primId, hitPos, normal);
                        albedo = getSceneArchetypeAlbedo(pc.sceneType, matId);
                    }
                    float3 sunDir, sunColor;
                    getSceneSunParams(pc.sceneType, sunDir, sunColor);
                    float NdotL = max(0.0, dot(normal, sunDir));
                    float3 color = albedo * 0.15 + albedo * sunColor * NdotL;
                    if (pc.dumpRenders != 0u) {
                        fb[pixelIdx] = float4(tonemapACES(color), 1.0);
                    } else {
                        threadLum = luminanceToUint(color);
                    }
                }
            } else if (mode == 9u) {
                assignedQueue = 0u;
                payload0 = float4(t, asfloat(pixelIdx), asfloat(primId), asfloat(packUnorm2x16_spv(rawBary)));
            } else if (mode == 0u) {
                if (pc.isGltf != 0u) {
                    uint matId = triMatBuf[primId];
                    if (WaveActiveAllEqual(matId)) {
                        matId = WaveReadLaneFirst(matId);
                    }
                    GltfMaterialGpu mat = matBuf[matId];
                    assignedQueue = mat.archetype % 8u;
                    float2 bary = unpackUnorm2x16_spv(packUnorm2x16_spv(rawBary));
                    float3 interpHitPos, geomNormal, smoothNormal;
                    float4 dummyTan;
                    float2 dummyUv;
                    unpackGltfHit(vbuf, primId, bary, false, interpHitPos, geomNormal, smoothNormal, dummyTan, dummyUv);
                    float3 faceNormal = (dot(smoothNormal, direction) > 0.0) ? -smoothNormal : smoothNormal;

                    uint visRng = pixelIdx;
                    float3 vis = computeGltfVisibility(topLevelAS, rq, mat, assignedQueue, interpHitPos, geomNormal, faceNormal, direction, pc.sceneType, true, visRng);

                    uint packedVis = packHalf2x16_spv(vis.xy);
                    payload0 = float4(asfloat(packedVis), asfloat(pixelIdx), asfloat(primId), asfloat(packUnorm2x16_spv(rawBary)));
                    payload1 = float4(vis.z, 0.0, 0.0, 0.0);
                } else {
                    float3 normal = computeSmoothNormal(vbuf, pc.sceneType, 0u, primId, hitPos, direction, float2(0.0));
                    assignedQueue = computeMaterialId(pc.sceneType, primId, hitPos, normal);
                    uint packedNorm = 0u;
                    if (pc.sceneType == SCENE_OUTDOOR_TERRAIN) {
                        packedNorm = packSnorm2x16_spv(octEncode(normal));
                        normal = octDecode(unpackSnorm2x16_spv(packedNorm));
                    }
                    uint visRng = pixelIdx;
                    float shadowVis;
                    uint3 aux;
                    computeProceduralVisibility(topLevelAS, rq, pc.sceneType, assignedQueue, hitPos, normal, direction, true, visRng, shadowVis, aux);
                    if (pc.sceneType == SCENE_OUTDOOR_TERRAIN) {
                        payload0 = float4(t, asfloat(pixelIdx), asfloat(packedNorm), shadowVis);
                    } else {
                        payload0 = float4(t, asfloat(pixelIdx), asfloat(primId), shadowVis);
                        payload1 = asfloat(uint4(aux.x, aux.y, aux.z, 0u));
                    }
                }
            } else {
                // Mode 1 (Multi-Bounce Path Tracing) & Mode 2 (Incoherent Secondary GI)
                float2 bary = rawBary;
                float3 normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, primId, hitPos, direction, bary);
                if (mode == 1u) {
                    assignedQueue = 0u;

                    float3 albedo = float3(0.72);
                    float3 emissive = float3(0.0);
                    float metallic = 0.0;
                    float roughness = 0.5;
                    float transmission = 0.0;
                    uint archetype = 0u;

                    if (pc.isGltf != 0u) {
                        uint matId = triMatBuf[primId];
                        if (WaveActiveAllEqual(matId)) {
                            matId = WaveReadLaneFirst(matId);
                        }
                        GltfMaterialGpu mat = matBuf[matId];
                        albedo = mat.baseColorFactor.rgb;
                        if (mat.baseColorTexIdx >= 0 || mat.emissiveTexIdx >= 0) {
                            float2 ubary = unpackUnorm2x16_spv(packUnorm2x16_spv(bary));
                            float2 uv = unpackGltfUV(vbuf, primId, ubary);
                            if (mat.baseColorTexIdx >= 0) {
                                albedo *= pow(sampleTexture(texHeaders, texPixels, uint(mat.baseColorTexIdx), uv).rgb, float3(2.2));
                            }
                            if (mat.emissiveTexIdx >= 0) {
                                emissive = pow(sampleTexture(texHeaders, texPixels, uint(mat.emissiveTexIdx), uv).rgb, float3(2.2)) * mat.emissiveFactor.rgb;
                            }
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

                    float3 primaryRadiance = emissive;
                    float3 throughput = float3(0.0);
                    float3 bOrigin = float3(0.0);
                    uint packedDir = 0u;

                    if (transmission > 0.5 || archetype == 2u) {
                        float3 cabinInterior = float3(0.02, 0.025, 0.03);
                        float3 R = reflect(direction, normal);
                        float3 envRefl = evalStudioEnvironment(R);
                        float cosTheta = clamp(dot(-direction, normal), 0.0, 1.0);
                        float F = 0.04 + 0.96 * pow(1.0 - cosTheta, 5.0);
                        primaryRadiance += lerp(cabinInterior, envRefl, F);
                        assignedQueue = 0xFFFFFFFFu;
                    } else {
                        if (pc.sceneType == SCENE_SHOWROOM_STUDIO) {
                            for (int i = 0; i < 3; ++i) {
                                float3 sDir, sCol;
                                getStudioLight(i, sDir, sCol);
                                float sNdotL = max(0.0, dot(normal, sDir));
                                if (sNdotL > 0.0) {
                                    if (!traceShadowRay(topLevelAS, rq, hitPos, normal, sDir, 5000.0)) {
                                        if (pc.isGltf != 0u) {
                                            primaryRadiance += evaluateCookTorranceGGX(albedo, roughness, metallic, normal, -direction, sDir, sCol);
                                        } else {
                                            primaryRadiance += (albedo / 3.14159) * sCol * sNdotL;
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
                                        primaryRadiance += evaluateCookTorranceGGX(albedo, roughness, metallic, normal, -direction, sunDir, sunColor);
                                    } else {
                                        primaryRadiance += (albedo / 3.14159) * sunColor * NdotL;
                                    }
                                }
                            }
                        }

                        if (pc.sceneType == SCENE_INDOOR_ATRIUM) {
                            for (int li = 0; li < 3; ++li) {
                                float3 lPos, lCol;
                                getAtriumLantern(li, lPos, lCol);
                                float3 toL = lPos - hitPos;
                                float lNdotToL = dot(normal, toL);
                                if (lNdotToL > 0.0) {
                                    float d2 = dot(toL, toL);
                                    float d = sqrt(d2);
                                    float invD = 1.0 / max(d, 0.001);
                                    float3 L = toL * invD;
                                    if (!traceShadowRay(topLevelAS, rq, hitPos, normal, L, d - 0.1)) {
                                        float lNdotL = lNdotToL * invD;
                                        float atten = 1.0 / (1.0 + 0.000004 * d2);
                                        primaryRadiance += (albedo / 3.14159) * lCol * atten * lNdotL;
                                    }
                                }
                            }
                        }

                        float u1 = randFloat(rngState);
                        float u2 = randFloat(rngState);
                        float3 bDir;
                        if (metallic > 0.1 && u2 < metallic) {
                            float3 refl = reflect(direction, normal);
                            float3 jitterVec = (float3(u1, u2, frac(u1 * 31.7)) * 2.0 - 1.0) * (roughness * 0.4);
                            bDir = normalize(refl + jitterVec);
                            if (dot(bDir, normal) <= 0.0) bDir = refl;
                            throughput = lerp(float3(0.04), albedo, metallic);
                        } else {
                            bDir = alignToNormal(cosineSampleHemisphere(u1, u2), normal);
                            throughput = (1.0 - metallic) * albedo;
                        }
                        float rayOffset = (pc.isGltf != 0u) ? 0.05 : 0.005;
                        bOrigin = hitPos + normal * rayOffset;
                        assignedQueue = computeBounceOctant(pc.sceneType, bDir);
                        packedDir = packSnorm2x16_spv(octEncode(bDir));
                    }

                    if (pc.dumpRenders == 0u) {
                        threadLum = luminanceToUint(primaryRadiance);
                    }

                    // Primary + Bounce 1 fused pass
                    if (assignedQueue < 8u) {
                        float rayOffset = (pc.isGltf != 0u) ? 0.05 : 0.005;
                        float tMaxBounce = (pc.isGltf != 0u) ? 5000.0 : ((pc.sceneType == SCENE_OUTDOOR_TERRAIN) ? 2500.0 : 50.0);
                        threadRays += 1u;
                        float3 b1Dir = octDecode(unpackSnorm2x16_spv(packedDir));

                        bool bHit = false;
                        float b_t = 0.0;
                        uint b_primId = 0u;
                        float2 b_bary = float2(0.0);
                        {
                            RayDesc bRay = { bOrigin, rayOffset, b1Dir, tMaxBounce };
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
                            threadHits += 1u;
                            float3 b_hitPos = bOrigin + b1Dir * b_t;
                            if (WaveActiveAllEqual(b_primId)) {
                                b_primId = WaveReadLaneFirst(b_primId);
                            }
                            float3 b_normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, b_primId, b_hitPos, b1Dir, b_bary);
                            float3 b_albedo = float3(0.72);
                            if (pc.isGltf != 0u) {
                                uint b_matId = triMatBuf[b_primId];
                                if (WaveActiveAllEqual(b_matId)) {
                                    b_matId = WaveReadLaneFirst(b_matId);
                                }
                                b_albedo = matBuf[b_matId].baseColorFactor.rgb;
                                int b_baseTex = matBuf[b_matId].baseColorTexIdx;
                                if (b_baseTex >= 0) {
                                    float2 b_ubary = unpackUnorm2x16_spv(packUnorm2x16_spv(b_bary));
                                    float2 b_uv = unpackGltfUV(vbuf, b_primId, b_ubary);
                                    b_albedo *= pow(sampleTexture(texHeaders, texPixels, uint(b_baseTex), b_uv).rgb, float3(2.2));
                                }
                            } else {
                                uint b_matId = computeMaterialId(pc.sceneType, b_primId, b_hitPos, b_normal);
                                b_albedo = getSceneArchetypeAlbedo(pc.sceneType, b_matId);
                            }
                            throughput *= b_albedo;

                            float3 b_rad = evalPathBounceDirectLight(topLevelAS, rq, pc.sceneType, pc.isGltf, b_hitPos, b_normal, b_albedo, false);
                            float3 b1Indirect = throughput * b_rad;
                            if (pc.dumpRenders != 0u) {
                                primaryRadiance += b1Indirect;
                            } else {
                                threadLum += luminanceToUint(b1Indirect);
                            }

                            if (pc.bounces >= 2u) {
                                uint bRng = pixelIdx + pc.seed + 1337u;
                                float bu1 = randFloat(bRng);
                                float bu2 = randFloat(bRng);
                                float3 nextBDir = alignToNormal(cosineSampleHemisphere(bu1, bu2), b_normal);
                                uint nextRayOct = computeBounceOctant(pc.sceneType, nextBDir);
                                bOrigin = b_hitPos + b_normal * rayOffset;
                                packedDir = packSnorm2x16_spv(octEncode(nextBDir));
                                assignedQueue = 8u + nextRayOct;
                            } else {
                                assignedQueue = 0xFFFFFFFFu;
                            }
                        } else {
                            float3 b1Indirect = throughput * evalSky(pc.sceneType, b1Dir);
                            if (pc.dumpRenders != 0u) {
                                primaryRadiance += b1Indirect;
                            } else {
                                threadLum += luminanceToUint(b1Indirect);
                            }
                            assignedQueue = 0xFFFFFFFFu;
                        }
                    }

                    payload0 = float4(bOrigin, asfloat(packedDir));
                    payload1 = float4(throughput, asfloat(pixelIdx));

                    if (pc.dumpRenders != 0u) {
                        if (assignedQueue == 0xFFFFFFFFu) {
                            float expGain = (pc.sceneType == SCENE_INDOOR_ATRIUM) ? 1.6 : 1.0;
                            if (pc.spp <= 1u) {
                                fb[pixelIdx] = float4(tonemapACES(primaryRadiance * expGain), 1.0);
                            } else if (pc.bounce == 0u) {
                                fb[pixelIdx] = float4(primaryRadiance, 1.0);
                            } else if (pc.bounce < pc.spp - 1u) {
                                fb[pixelIdx] = fb[pixelIdx] + float4(primaryRadiance, 0.0);
                            } else {
                                float3 total = fb[pixelIdx].rgb + primaryRadiance;
                                float3 avg = (total / float(pc.spp)) * expGain;
                                fb[pixelIdx] = float4(tonemapACES(avg), 1.0);
                            }
                        } else {
                            if (pc.bounce == 0u) {
                                fb[pixelIdx] = float4(primaryRadiance, 1.0);
                            } else {
                                fb[pixelIdx] = fb[pixelIdx] + float4(primaryRadiance, 0.0);
                            }
                        }
                    }
                } else {
                    float u1 = randFloat(rngState);
                    float u2 = randFloat(rngState);
                    float3 incohDir = alignToNormal(cosineSampleHemisphere(u1, u2), normal);
                    float rayOffset = (pc.isGltf != 0u) ? 0.05 : 0.005;
                    uint oct8 = (incohDir.x >= 0.0 ? 1u : 0u)
                              | (incohDir.y >= 0.0 ? 2u : 0u)
                              | (incohDir.z >= 0.0 ? 4u : 0u);
                    uint oct16 = oct8 | ((abs(incohDir.z) >= 0.6) ? 8u : 0u);
                    assignedQueue = oct16;
                    uint packedDir = packSnorm2x16_spv(octEncode(incohDir));
                    payload0 = float4(hitPos + normal * rayOffset, asfloat(packedDir));
                }
            }
        } else {
            if (pc.dumpRenders != 0u) {
                if (mode == 5u) {
                    fb[pixelIdx] = float4(0.02, 0.03, 0.05, 1.0);
                } else if (mode == 8u) {
                    float3 skyCol = evalSky(pc.sceneType, direction) * 0.15;
                    fb[pixelIdx] = float4(tonemapACES(skyCol), 1.0);
                } else if (mode == 9u) {
                    fb[pixelIdx] = float4(0.04, 0.05, 0.08, 1.0);
                } else if (mode == 1u) {
                    float3 sampleRadiance = evalSky(pc.sceneType, direction);
                    float expGain = (pc.sceneType == SCENE_INDOOR_ATRIUM) ? 1.6 : 1.0;
                    if (pc.spp <= 1u) {
                        fb[pixelIdx] = float4(tonemapACES(sampleRadiance * expGain), 1.0);
                    } else if (pc.bounce == 0u) {
                        fb[pixelIdx] = float4(sampleRadiance, 1.0);
                    } else if (pc.bounce < pc.spp - 1u) {
                        fb[pixelIdx] = fb[pixelIdx] + float4(sampleRadiance, 0.0);
                    } else {
                        float3 total = fb[pixelIdx].rgb + sampleRadiance;
                        float3 avg = (total / float(pc.spp)) * expGain;
                        fb[pixelIdx] = float4(tonemapACES(avg), 1.0);
                    }
                } else {
                    float3 skyCol = evalSky(pc.sceneType, direction);
                    fb[pixelIdx] = float4(tonemapACES(skyCol), 1.0);
                }
            } else if (mode == 0u || mode == 1u) {
                float3 skyCol = evalSky(pc.sceneType, direction);
                threadLum = luminanceToUint(skyCol);
            } else if (mode == 8u) {
                float3 skyCol = evalSky(pc.sceneType, direction) * 0.15;
                threadLum = luminanceToUint(skyCol);
            }
        }
    }

    // Coalesced Zero-LDS Wave-Level Stream Compaction into Contiguous VRAM Queues
    if (mode == 6u || mode == 7u) {
        bool match = (assignedQueue == 0u);
        uint slotInWave = WavePrefixCountBits(match);
        uint count = WaveActiveCountBits(match);
        uint4 ballot = WaveActiveBallot(match);
        if (count > 0u) {
            uint firstLane = ballotFindLSB(ballot);
            uint baseSlot = 0u;
            if (WaveGetLaneIndex() == firstLane) {
                baseSlot = wlAtomicAddCounter(worklist, 0u, count);
            }
            baseSlot = WaveReadLaneAt(baseSlot, firstLane);
            if (match) {
                uint slot = baseSlot + slotInWave;
                if (mode == 6u && pc.bounce == 1u) {
                    if (slot < pc.queueCapacity) {
                        uint baseIdx = slot * 8u;
                        wlStoreRecord(worklist, baseIdx + 0u, payload0);
                        wlStoreRecord(worklist, baseIdx + 1u, payload1);
                        wlStoreRecord(worklist, baseIdx + 2u, float4(payload0.xyz, 1.0));
                        wlStoreRecord(worklist, baseIdx + 3u, float4(payload1.xy, payload0.w, 0.0));
                        wlStoreRecord(worklist, baseIdx + 4u, float4(0.72, 0.72, 0.72, 1.0));
                        wlStoreRecord(worklist, baseIdx + 5u, float4(0.5, 0.0, 1.5, 0.0));
                        wlStoreRecord(worklist, baseIdx + 6u, payload0);
                        wlStoreRecord(worklist, baseIdx + 7u, payload1);
                    }
                } else {
                    if (slot < pc.queueCapacity) {
                        wlStoreRecord(worklist, slot * 2u, payload0);
                        wlStoreRecord(worklist, slot * 2u + 1u, payload1);
                    }
                }
            }
        }
    } else if (mode == 9u || (mode == 8u && pc.spp <= 1u) || (mode == 5u && pc.spp != 3u)) {
        bool match = (assignedQueue == 0u);
        uint slotInWave = WavePrefixCountBits(match);
        uint count = WaveActiveCountBits(match);
        uint4 ballot = WaveActiveBallot(match);
        if (count > 0u) {
            uint firstLane = ballotFindLSB(ballot);
            uint baseSlot = 0u;
            if (WaveGetLaneIndex() == firstLane) {
                baseSlot = wlAtomicAddCounter(worklist, 0u, count);
            }
            baseSlot = WaveReadLaneAt(baseSlot, firstLane);
            if (match) {
                uint slot = baseSlot + slotInWave;
                if (slot < pc.queueCapacity) {
                    wlStoreRecord(worklist, slot, payload0);
                }
            }
        }
    } else {
        uint maxQ = (mode == 2u || mode == 1u) ? 16u : 8u;
        if (mode == 5u && pc.spp == 3u) {
            maxQ = 3u;
        }
        bool hasValidQueue = (assignedQueue < maxQ);
        uint waveOctMask = WaveActiveBitOr(hasValidQueue ? (1u << assignedQueue) : 0u);
        if (waveOctMask != 0u) {
            uint mySlotInWave = 0u;
            uint qCount = 0u;
            uint maskCopy = waveOctMask;
            uint laneId = WaveGetLaneIndex();
            while (maskCopy != 0u) {
                uint q = firstbitlow(maskCopy);
                maskCopy &= maskCopy - 1u;
                bool isQ = (assignedQueue == q);
                uint slotQ = WavePrefixCountBits(isQ);
                uint cntQ = WaveActiveCountBits(isQ);
                if (isQ) {
                    mySlotInWave = slotQ;
                }
                if (laneId == q) {
                    qCount = cntQ;
                }
            }
            uint qBase = 0u;
            if (qCount > 0u) {
                qBase = wlAtomicAddCounter(worklist, laneId, qCount);
            }
            uint myBase = WaveReadLaneAt(qBase, min(assignedQueue, 15u));
            if (hasValidQueue) {
                uint slot = myBase + mySlotInWave;
                if (slot < pc.queueCapacity) {
                    if (mode == 1u) {
                        uint recIdx = (assignedQueue * pc.queueCapacity + slot) * 2u;
                        wlStoreRecord(worklist, recIdx, payload0);
                        wlStoreRecord(worklist, recIdx + 1u, payload1);
                    } else {
                        wlStoreRecord(worklist, assignedQueue * pc.queueCapacity + slot, payload0);
                        if (mode == 0u && ((pc.isGltf != 0u && pc.sceneType == SCENE_AAA_FOREST && assignedQueue == 5u) || (pc.isGltf == 0u && pc.sceneType != SCENE_OUTDOOR_TERRAIN))) {
                            wlStoreRecord(worklist, (8u + assignedQueue) * pc.queueCapacity + slot, payload1);
                        }
                    }
                }
            }
        }
    }

    addWaveCounters(results, threadRays, threadHits, threadLum);
}
