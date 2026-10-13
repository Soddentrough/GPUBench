#version 460
#extension GL_EXT_ray_tracing : require

struct Payload16B {
    vec4 data;
};

layout(location = 0) rayPayloadInEXT Payload16B payload;

void main() {
    payload.data = payload.data * 0.25 + vec4(gl_WorldRayDirectionEXT, gl_RayTmaxEXT * 0.001);
}
