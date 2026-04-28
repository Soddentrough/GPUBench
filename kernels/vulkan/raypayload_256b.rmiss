#version 460
#extension GL_EXT_ray_tracing : require

struct Payload256B {
    vec4 data[16];
};

layout(location = 0) rayPayloadInEXT Payload256B payload;

void main() {
    payload.data[0] = vec4(0.0);
}
