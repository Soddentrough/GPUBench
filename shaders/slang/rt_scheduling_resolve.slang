import rt_types;

[[vk::binding(0, 0)]] RWStructuredBuffer<uint4> worklist;
[[vk::binding(1, 0)]] RWStructuredBuffer<VkDispatchIndirectCommand> indirectCmds;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> dgcCount;
[[vk::binding(3, 0)]] RWStructuredBuffer<DGCSequenceItem> dgcData; // 0..31: templates, 32..63: compacted

struct ResolvePC {
    uint resetQueue;
    uint dgcMode; // 0: legacy, 1: standard DGC, 2: consolidated octant, 3: ping-pong bounce, 4: shadows, 5: single light, 6: 128 lights, 7: G-buffer normals
    uint bounceIndex;
};

[[vk::push_constant]] ResolvePC rpc;

[numthreads(32, 1, 1)]
void main(uint3 lid : SV_GroupThreadID) {
    uint id = lid.x;
    uint bank = (rpc.bounceIndex & 1u);
    uint baseQ = bank * 8u;
    bool pingPong = (rpc.dgcMode == 3u || rpc.dgcMode == 7u) || (rpc.resetQueue != 0xFFFFFFFFu) || (rpc.bounceIndex > 0u);

    if (id < 16u && (!pingPong || (id >= baseQ && id < baseQ + 8u))) {
        uint count = wlLoadCounter(worklist, id);
        indirectCmds[id].x = (count + 31u) / 32u;
        indirectCmds[id].y = (count > 0u) ? 1u : 0u;
        indirectCmds[id].z = (count > 0u) ? 1u : 0u;
        wlStoreSnapshot(worklist, id, count);
        wlStoreFastSnap(worklist, id, count);
        wlStoreCounter(worklist, id, 0u);
    } else if (pingPong && rpc.resetQueue == 0xFFFFFFFFu && id < 16u) {
        wlStoreCounter(worklist, id, 0u);
    }

    if (rpc.dgcMode == 2u) {
        uint waves = (id < 16u) ? indirectCmds[id].x : 0u;
        uint pref = WavePrefixSum(waves);
        if (id < 16u) {
            wlStorePrefix(worklist, id, pref);
        }
        if (id == 15u) {
            uint totalWaves = pref + waves;
            indirectCmds[8].x = totalWaves;
            indirectCmds[8].y = (totalWaves > 0u) ? 1u : 0u;
            indirectCmds[8].z = (totalWaves > 0u) ? 1u : 0u;
        }
    } else {
        uint waves = (id < 8u) ? indirectCmds[baseQ + id].x : 0u;
        uint pref = WavePrefixSum(waves);
        if (id < 8u) {
            wlStorePrefix(worklist, baseQ + id, pref);
        }
        if (id == 7u) {
            uint totalWaves = pref + waves;
            if (bank == 0u) {
                indirectCmds[8].x = totalWaves;
                indirectCmds[8].y = (totalWaves > 0u) ? 1u : 0u;
                indirectCmds[8].z = (totalWaves > 0u) ? 1u : 0u;
            }
            indirectCmds[16u + bank].x = totalWaves;
            indirectCmds[16u + bank].y = (totalWaves > 0u) ? 1u : 0u;
            indirectCmds[16u + bank].z = (totalWaves > 0u) ? 1u : 0u;
        }
    }

    if (rpc.dgcMode == 1u) {
        bool isActive = false;
        uint count = 0u;
        if (id < 8u) {
            count = wlLoadSnapshot(worklist, id);
            isActive = (count > 0u);
            dgcData[id].cmdX = (count + 31u) / 32u;
            dgcData[id].cmdY = isActive ? 1u : 0u;
            dgcData[id].cmdZ = isActive ? 1u : 0u;
        }

        uint activeMask = isActive ? 1u : 0u;
        uint slot = WavePrefixSum(activeMask);
        uint totalActive = WaveActiveSum(activeMask);

        if (isActive) {
            DGCSequenceItem item = dgcData[id];
            item.cmdX = (count + 31u) / 32u;
            item.cmdY = 1u;
            item.cmdZ = 1u;
            dgcData[32u + slot] = item;
        }

        if (id == 0u) {
            dgcCount[0] = totalActive;
        }
    } else if (rpc.dgcMode == 2u) {
        if (id == 7u) {
            uint totalWaves = indirectCmds[8].x;
            if (totalWaves > 0u) {
                DGCSequenceItem item = dgcData[8];
                item.cmdX = totalWaves;
                item.cmdY = 1u;
                item.cmdZ = 1u;
                dgcData[32] = item;
                dgcCount[0] = 1u;
            } else {
                dgcCount[0] = 0u;
            }
        }
    } else if (rpc.dgcMode == 3u) {
        if (id == 7u) {
            uint totalWaves = indirectCmds[16u + bank].x;
            if (totalWaves > 0u) {
                uint templateIdx = 16u + rpc.bounceIndex;
                DGCSequenceItem item = dgcData[templateIdx];
                uint sampleSeed = wlLoadSeed(worklist);
                item.pc[3] = ((sampleSeed != 0u) ? sampleSeed : item.pc[3]) + rpc.bounceIndex * 7919u;
                item.cmdX = totalWaves;
                item.cmdY = 1u;
                item.cmdZ = 1u;
                dgcData[32] = item;
                dgcCount[0] = 1u;
            } else {
                dgcCount[0] = 0u;
            }
        }
    } else if (rpc.dgcMode == 4u) {
        bool isActive = false;
        uint count = 0u;
        if (id < 3u) {
            count = wlLoadSnapshot(worklist, id);
            isActive = (count > 0u);
        }

        uint activeMask = isActive ? 1u : 0u;
        uint slot = WavePrefixSum(activeMask);
        uint totalActive = WaveActiveSum(activeMask);

        if (isActive) {
            DGCSequenceItem item = dgcData[9u + id];
            item.cmdX = (count + 31u) / 32u;
            item.cmdY = 1u;
            item.cmdZ = 1u;
            dgcData[32u + slot] = item;
        }

        if (id == 0u) {
            dgcCount[0] = totalActive;
        }
    } else if (rpc.dgcMode == 5u) {
        if (id == 0u) {
            uint count = wlLoadSnapshot(worklist, 0u);
            indirectCmds[0].x = (count + 31u) / 32u;
            indirectCmds[0].y = (count > 0u) ? 1u : 0u;
            indirectCmds[0].z = (count > 0u) ? 1u : 0u;

            if (count > 0u) {
                DGCSequenceItem item = dgcData[12];
                item.cmdX = (count + 31u) / 32u;
                item.cmdY = 1u;
                item.cmdZ = 1u;
                dgcData[32] = item;
                dgcCount[0] = 1u;
            } else {
                dgcCount[0] = 0u;
            }
        }
    } else if (rpc.dgcMode == 6u) {
        bool isActive = false;
        uint count = 0u;
        if (id < 16u) {
            count = wlLoadSnapshot(worklist, id);
            isActive = (count > 0u);
        }

        uint activeMask = isActive ? 1u : 0u;
        uint slot = WavePrefixSum(activeMask);
        uint totalActive = WaveActiveSum(activeMask);

        if (isActive) {
            DGCSequenceItem item = dgcData[13];
            item.pc[0] = id;
            item.cmdX = (count + 31u) / 32u;
            item.cmdY = 1u;
            item.cmdZ = 1u;
            dgcData[32u + slot] = item;
        }

        if (id == 0u) {
            dgcCount[0] = totalActive;
        }
    } else if (rpc.dgcMode == 7u) {
        if (id == 7u) {
            uint totalWaves = indirectCmds[16u + bank].x;
            if (totalWaves > 0u) {
                uint templateIdx = 24u + rpc.bounceIndex;
                DGCSequenceItem item = dgcData[templateIdx];
                item.cmdX = totalWaves;
                item.cmdY = 1u;
                item.cmdZ = 1u;
                dgcData[32] = item;
                dgcCount[0] = 1u;
            } else {
                dgcCount[0] = 0u;
            }
        }
    }
}
