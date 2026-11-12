#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload {
    vec3 color;
    uint hits;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;
hitAttributeEXT vec2 attribs;

void main() {
    float b0 = 1.0 - attribs.x - attribs.y;
    vec3 leafTint = vec3(0.18 + 0.25 * attribs.x,
                         0.55 + 0.30 * attribs.y,
                         0.14 + 0.15 * b0);
    float depthAtten = 1.0 / (1.0 + 0.08 * gl_HitTEXT);
    payload.color = leafTint * depthAtten;
    payload.hits = uint(gl_PrimitiveID) + 1u;
}
