#version 460
#extension GL_EXT_ray_tracing : require

struct Payload16B {
    vec4 data;
};

layout(location = 0) rayPayloadInEXT Payload16B payload;

void main() {
    payload.data = vec4(0.0);
}
