#version 460
#extension GL_EXT_ray_tracing : require

struct Payload16B {
    vec4 data;
};

layout(location = 0) rayPayloadInEXT Payload16B payload;
hitAttributeEXT vec2 attribs;

void main() {
    payload.data = vec4(1.0);
}
