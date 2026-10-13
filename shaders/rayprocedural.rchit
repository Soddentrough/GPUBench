#version 460
#extension GL_EXT_ray_tracing : require

struct SphereHit {
    vec3 normal;
};

hitAttributeEXT SphereHit hitAttr;
layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    payload = hitAttr.normal * 0.5 + vec3(1.0);
}
