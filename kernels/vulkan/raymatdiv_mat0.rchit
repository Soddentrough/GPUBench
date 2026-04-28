#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 attribs;

void main() {
    // Mimic simple diffuse material
    float nDotL = max(dot(vec3(0.0, 1.0, 0.0), vec3(0.5, 0.5, 0.5)), 0.0);
    payload = vec3(1.0, 0.0, 0.0) * nDotL;
}
