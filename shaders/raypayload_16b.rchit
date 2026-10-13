#version 460
#extension GL_EXT_ray_tracing : require

struct Payload16B {
    vec4 data;
};

layout(location = 0) rayPayloadInEXT Payload16B payload;
hitAttributeEXT vec2 attribs;

void main() {
    float t = gl_HitTEXT;
    float p = float(gl_PrimitiveID & 255);
    payload.data = payload.data * 0.5 + vec4(t, attribs.x, attribs.y, p * 0.01 + 1.0);
}
