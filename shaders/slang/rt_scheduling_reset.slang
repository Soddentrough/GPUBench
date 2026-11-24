import rt_types;

[[vk::binding(0, 0)]] RWStructuredBuffer<uint4> worklist;
[[vk::binding(1, 0)]] RWStructuredBuffer<VkDispatchIndirectCommand> indirectCmds;

[numthreads(32, 1, 1)]
void main(uint3 lid : SV_GroupThreadID) {
    uint id = lid.x;
    if (id < 16u) {
        wlStoreCounter(worklist, id, 0u);
        wlStoreSnapshot(worklist, id, 0u);
        wlStoreFastSnap(worklist, id, 0u);
        wlStorePrefix(worklist, id, 0u);
    }
    if (id < 32u) {
        indirectCmds[id].x = 0u;
        indirectCmds[id].y = 1u;
        indirectCmds[id].z = 1u;
    }
}
