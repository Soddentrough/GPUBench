#version 460
#extension GL_EXT_ray_tracing : require

struct Payload256B {
    vec4 data[16];
};

layout(location = 0) rayPayloadInEXT Payload256B payload;
hitAttributeEXT vec2 attribs;

void main() {
    payload.data[0] = vec4(1.0);
    // Force usage of the payload data
    for (int i = 1; i < 16; ++i) {
        payload.data[i] = vec4(1.0);
    }
}
