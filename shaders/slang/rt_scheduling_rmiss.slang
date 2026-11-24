import rt_types;
import rt_scene;

[[vk::constant_id(0)]] const uint RMISS_MODE = 7u;

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

[shader("miss")]
void main(inout RayPayload payload) {
    if (RMISS_MODE == 7u || pc.mode == 7u) {
        payload.color = float3(-1.0);
        return;
    }
    float3 sky = evalSky(pc.sceneType, WorldRayDirection());
    payload.color = -max(sky, float3(1e-4));
}
