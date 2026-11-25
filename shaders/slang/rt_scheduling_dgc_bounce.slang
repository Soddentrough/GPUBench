import rt_types;
import rt_math;
import rt_scene;
import rt_pbr;

[[vk::constant_id(0)]] const uint BOUNCE_MODE = 0u; // 0: Non-terminal PT, 1: Terminal PT, 2: Non-terminal GI, 3: Terminal GI

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] RWStructuredBuffer<Results> results;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint4> worklist;
[[vk::binding(3, 0)]] StructuredBuffer<float> vbuf;
[[vk::binding(4, 0)]] StructuredBuffer<GltfMaterialGpu> matBuf;
[[vk::binding(5, 0)]] StructuredBuffer<uint> triMatBuf;
[[vk::binding(6, 0)]] StructuredBuffer<TextureHeaderGpu> texHeaders;
[[vk::binding(7, 0)]] StructuredBuffer<uint> texPixels;
[[vk::binding(8, 0)]] RWStructuredBuffer<float4> fb;

struct PushConstants {
    uint inQueue;
    uint outQueue;
    uint bounceIndex;
    uint seed;
    uint maxQueueSize;
    uint sceneType;
    uint isGltf;
};

[[vk::push_constant]] PushConstants pc;

[numthreads(32, 1, 1)]
void main(uint3 wgid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    uint waveId = wgid.x;
    uint lid = ltid.x;
    bool isGi = (BOUNCE_MODE >= 2u);
    uint baseQ = (pc.inQueue & 1u) * 8u;
    uint numQ = 8u;
    uint qPref = (lid < numQ) ? wlLoadPrefix(worklist, baseQ + lid) : 0xFFFFFFFFu;
    uint qSnap = (lid < numQ) ? wlLoadFastSnap(worklist, baseQ + lid) : 0u;
    uint4 geMask = WaveActiveBallot(waveId >= qPref);
    uint myOct = (geMask.x != 0u) ? firstbithigh(geMask.x) : 0u;
    uint oct = baseQ + myOct;
    uint waveOffsetInOct = waveId - WaveReadLaneAt(qPref, myOct);
    uint activeCount = min(WaveReadLaneAt(qSnap, myOct), pc.maxQueueSize);

    uint sampleSeed = 0u;
    uint packedMeta = 0u;
    if (!isGi) {
        if (lid == 0u) {
            sampleSeed = wlLoadSeed(worklist);
            packedMeta = wlLoadMeta(worklist);
        }
        sampleSeed = WaveReadLaneFirst(sampleSeed);
        packedMeta = WaveReadLaneFirst(packedMeta);
    }

    uint rayIdx = waveOffsetInOct * 32u + lid;
    bool activeRay = (rayIdx < activeCount);
    uint threadRays = activeRay ? 1u : 0u;
    uint threadHits = 0u;
    uint threadLum = 0u;

    uint nextRayOct = 0xFFFFFFFFu;
    float4 nextRayPayload = float4(0.0);
    float4 nextRayPayload1 = float4(0.0);

    uint dumpRenders = packedMeta & 0xFFu;
    uint totalSpp = (packedMeta >> 8u) & 0xFFu;
    uint sampleIdx = (packedMeta >> 16u) & 0xFFu;

    if (activeRay) {
        RayQuery<RAY_FLAG_NONE> rq;
        uint recordIdx = isGi
            ? (oct * pc.maxQueueSize + rayIdx)
            : ((oct * pc.maxQueueSize + rayIdx) * 2u);
        float4 p = wlLoadRecord(worklist, recordIdx);
        float3 origin = p.xyz;
        float3 direction = octDecode(unpackSnorm2x16_spv(asuint(p.w)));

        float tMaxBounce = isGi
            ? ((pc.isGltf != 0u) ? 5000.0 : ((pc.sceneType == SCENE_OUTDOOR_TERRAIN) ? 500.0 : 20.0))
            : ((pc.isGltf != 0u) ? 5000.0 : ((pc.sceneType == SCENE_OUTDOOR_TERRAIN) ? 2500.0 : 50.0));
        float rayOffset = (pc.isGltf != 0u) ? 0.05 : 0.005;

        bool hitFound = false;
        float b_t = 0.0;
        uint b_primId = 0u;
        float2 b_bary = float2(0.0);
        {
            RayDesc ray = { origin, rayOffset, direction, tMaxBounce };
            rq.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, ray);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                hitFound = true;
                b_t = rq.CommittedRayT();
                b_primId = rq.CommittedPrimitiveIndex();
                b_bary = rq.CommittedTriangleBarycentrics();
            }
        }

        if (isGi) {
            if (hitFound) {
                threadHits = 1u;
                if (BOUNCE_MODE == 2u && pc.outQueue < 32u) {
                    uint h = giRayHash(origin, asuint(p.w), pc.bounceIndex);
                    if (pc.bounceIndex < 2u || randFloat(h) >= 0.4) {
                        float3 b_hitPos = origin + direction * b_t;
                        if (WaveActiveAllEqual(b_primId)) {
                            b_primId = WaveReadLaneFirst(b_primId);
                        }
                        float3 b_normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, b_primId, b_hitPos, direction, b_bary);
                        float bu1 = randFloat(h);
                        float bu2 = randFloat(h);
                        float3 nextDir = alignToNormal(cosineSampleHemisphere(bu1, bu2), b_normal);
                        uint nextPackedDir = packSnorm2x16_spv(octEncode(nextDir));
                        float3 qNextDir = octDecode(unpackSnorm2x16_spv(nextPackedDir));
                        nextRayOct = computeBounceOctant(pc.sceneType, qNextDir);
                        nextRayPayload = float4(b_hitPos + b_normal * rayOffset, asfloat(nextPackedDir));
                    }
                }
            }
        } else {
            bool isTerminal = (BOUNCE_MODE != 0u) || (pc.outQueue >= 32u);
            float3 b_rad = float3(0.0);
            float3 b_hitPos = float3(0.0);
            float3 b_normal = float3(0.0, 1.0, 0.0);
            float3 b_albedo = float3(0.72);

            if (hitFound) {
                threadHits = 1u;
                b_hitPos = origin + direction * b_t;
                if (WaveActiveAllEqual(b_primId)) {
                    b_primId = WaveReadLaneFirst(b_primId);
                }
                b_normal = computeSmoothNormal(vbuf, pc.sceneType, pc.isGltf, b_primId, b_hitPos, direction, b_bary);

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

                b_rad = evalPathBounceDirectLight(topLevelAS, rq, pc.sceneType, pc.isGltf, b_hitPos, b_normal, b_albedo, true);
            } else {
                b_rad = evalSky(pc.sceneType, direction);
                isTerminal = true;
            }

            float4 p1 = wlLoadRecord(worklist, recordIdx + 1u);
            float3 throughput = p1.xyz;
            uint pixelIdx = asuint(p1.w);
            float3 indirectRadiance = throughput * b_rad;

            if (!isTerminal) {
                uint rngState = pixelIdx + sampleSeed + pc.bounceIndex * 1337u;
                bool survives = (pc.bounceIndex < 2u) || (randFloat(rngState) >= 0.4);
                if (survives) {
                    float u1 = randFloat(rngState);
                    float u2 = randFloat(rngState);
                    float3 nextDir = alignToNormal(cosineSampleHemisphere(u1, u2), b_normal);
                    nextRayOct = computeBounceOctant(pc.sceneType, nextDir);
                    nextRayPayload = float4(b_hitPos + b_normal * rayOffset, asfloat(packSnorm2x16_spv(octEncode(nextDir))));
                    nextRayPayload1 = float4(throughput * b_albedo, asfloat(pixelIdx));
                } else {
                    isTerminal = true;
                }
            }

            if (dumpRenders != 0u) {
                if (!isTerminal) {
                    fb[pixelIdx] = fb[pixelIdx] + float4(indirectRadiance, 0.0);
                } else {
                    float3 total = fb[pixelIdx].rgb + indirectRadiance;
                    float expGain = (pc.sceneType == SCENE_INDOOR_ATRIUM) ? 1.6 : 1.0;
                    if (totalSpp <= 1u) {
                        fb[pixelIdx] = float4(tonemapACES(total * expGain), 1.0);
                    } else if (sampleIdx < totalSpp - 1u) {
                        fb[pixelIdx] = float4(total, 1.0);
                    } else {
                        float3 avg = (total / float(totalSpp)) * expGain;
                        fb[pixelIdx] = float4(tonemapACES(avg), 1.0);
                    }
                }
            } else {
                threadLum = luminanceToUint(indirectRadiance);
            }
        }
    }

    // Ping-Pong Subgroup Compaction into 8 Directional Octant Queues of outQueue Bank
    if ((BOUNCE_MODE == 0u || BOUNCE_MODE == 2u) && pc.outQueue < 32u) {
        uint outBaseQ = (pc.outQueue & 1u) * 8u;
        bool hasNext = (nextRayOct < 8u);
        uint waveOctMask = WaveActiveBitOr(hasNext ? (1u << nextRayOct) : 0u);
        if (waveOctMask != 0u) {
            uint mySlotInWave = 0u;
            uint qCount = 0u;
            uint maskCopy = waveOctMask;
            uint laneId = WaveGetLaneIndex();
            while (maskCopy != 0u) {
                uint q = firstbitlow(maskCopy);
                maskCopy &= maskCopy - 1u;
                bool isQ = (nextRayOct == q);
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
                qBase = wlAtomicAddCounter(worklist, outBaseQ + laneId, qCount);
            }
            uint myBase = WaveReadLaneAt(qBase, min(nextRayOct, 7u));
            if (hasNext) {
                uint slot = myBase + mySlotInWave;
                if (slot < pc.maxQueueSize) {
                    if (BOUNCE_MODE == 2u) {
                        wlStoreRecord(worklist, (outBaseQ + nextRayOct) * pc.maxQueueSize + slot, nextRayPayload);
                    } else {
                        uint outRecIdx = ((outBaseQ + nextRayOct) * pc.maxQueueSize + slot) * 2u;
                        wlStoreRecord(worklist, outRecIdx, nextRayPayload);
                        wlStoreRecord(worklist, outRecIdx + 1u, nextRayPayload1);
                    }
                }
            }
        }
    }

    addWaveCounters(results, threadRays, threadHits, threadLum);
}
