#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 attribs;

void main() {
    // Mimic metallic material with some math
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 N = vec3(0.0, 1.0, 0.0);
    vec3 H = normalize(V + N);
    float NdotH = max(dot(N, H), 0.0);
    float spec = pow(NdotH, 32.0);
    payload = vec3(0.8, 0.8, 0.8) * spec;
}
