#version 460
#extension GL_EXT_ray_tracing : require

struct Payload256B {
    vec4 data[16];
};

layout(location = 0) rayPayloadInEXT Payload256B payload;
hitAttributeEXT vec2 attribs;

void main() {
    float t = gl_HitTEXT;
    float p = float(gl_PrimitiveID & 255);
    vec4 hitVec = vec4(t, attribs.x, attribs.y, p * 0.01 + 1.0);
    for (int i = 0; i < 16; ++i) {
        payload.data[i] = payload.data[i] * 0.5 + hitVec * float(i + 1);
    }
}
