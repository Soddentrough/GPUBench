#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 attribs;

void main() {
    // Mimic a noise-based procedural material
    vec2 p = attribs;
    float n = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    payload = vec3(0.0, 1.0, 0.0) * n;
}
