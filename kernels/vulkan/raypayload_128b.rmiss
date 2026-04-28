#version 460
#extension GL_EXT_ray_tracing : require

struct Payload128B {
    vec4 data[8];
};

layout(location = 0) rayPayloadInEXT Payload128B payload;

void main() {
    payload.data[0] = vec4(0.0);
}
