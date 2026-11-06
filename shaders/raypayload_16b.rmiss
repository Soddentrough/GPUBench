#version 460
#extension GL_EXT_ray_tracing : require

struct Payload16B {
    vec4 data;
};

layout(location = 0) rayPayloadInEXT Payload16B payload;

void main() {
    payload.data = -abs(payload.data) * 0.25 - vec4(abs(gl_WorldRayDirectionEXT), gl_RayTmaxEXT * 0.001 + 1.0);
}
