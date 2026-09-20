#version 460
#extension GL_EXT_ray_tracing : require

struct Payload128B {
    vec4 data[8];
};

layout(location = 0) rayPayloadInEXT Payload128B payload;
hitAttributeEXT vec2 attribs;

void main() {
    payload.data[0] = vec4(1.0);
    // Force usage of the payload data so compiler doesn't optimize it out
    payload.data[1] = vec4(1.0);
    payload.data[2] = vec4(1.0);
    payload.data[3] = vec4(1.0);
    payload.data[4] = vec4(1.0);
    payload.data[5] = vec4(1.0);
    payload.data[6] = vec4(1.0);
    payload.data[7] = vec4(1.0);
}
