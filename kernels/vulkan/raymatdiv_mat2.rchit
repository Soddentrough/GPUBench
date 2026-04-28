#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 attribs;

void main() {
    // Mimic glass material
    vec3 I = vec3(0.0, 0.0, 1.0);
    vec3 N = vec3(0.0, 1.0, 0.0);
    float eta = 1.5;
    float k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I));
    vec3 R = (k < 0.0) ? vec3(0.0) : eta * I - (eta * dot(N, I) + sqrt(k)) * N;
    payload = vec3(0.0, 0.0, 1.0) * R;
}
