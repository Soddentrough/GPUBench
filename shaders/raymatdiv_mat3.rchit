#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 attribs;

void main() {
    vec3 V = -normalize(gl_WorldRayDirectionEXT);
    vec3 N = normalize(vec3(attribs.x - 0.5, 1.0, attribs.y - 0.5));
    float cosV = abs(dot(N, V));
    float sheen = pow(1.0 - cosV, 4.0);
    vec3 color = vec3(0.1, 0.6, 0.2) + vec3(0.8, 0.3, 0.7) * sheen;
    payload = color;
}
