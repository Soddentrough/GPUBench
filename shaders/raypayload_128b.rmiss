#version 460
#extension GL_EXT_ray_tracing : require

struct Payload128B {
    vec4 data[8];
};

layout(location = 0) rayPayloadInEXT Payload128B payload;

void main() {
    vec4 missVec = vec4(gl_WorldRayDirectionEXT, gl_RayTmaxEXT * 0.001);
    for (int i = 0; i < 8; ++i) {
        payload.data[i] = payload.data[i] * 0.25 + missVec * float(i + 1);
    }
}
