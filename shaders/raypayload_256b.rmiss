#version 460
#extension GL_EXT_ray_tracing : require

struct Payload256B {
    vec4 data[16];
};

layout(location = 0) rayPayloadInEXT Payload256B payload;

void main() {
    vec4 missVec = vec4(gl_WorldRayDirectionEXT, gl_RayTmaxEXT * 0.001);
    for (int i = 0; i < 16; ++i) {
        payload.data[i] = payload.data[i] * 0.25 + missVec * float(i + 1);
    }
}
