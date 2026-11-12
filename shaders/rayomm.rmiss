#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload {
    vec3 color;
    uint hits;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main() {
    // Sky background
    payload.color = vec3(0.15, 0.28, 0.52);
    payload.hits = 0u;
}
