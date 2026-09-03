#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 attribs;

void main() {
    vec3 V = -normalize(gl_WorldRayDirectionEXT);
    vec3 N = normalize(vec3(attribs.x - 0.5, 1.0, attribs.y - 0.5));
    vec3 L = normalize(vec3(0.3, 0.8, 0.5));
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    vec3 baseDiffuse = vec3(0.8, 0.05, 0.05) * NdotL;
    vec3 H = normalize(V + L);
    float NdotH = clamp(dot(N, H), 0.0, 1.0);
    float coat = pow(NdotH, 64.0) * 0.25;
    payload = baseDiffuse * (1.0 - coat) + vec3(coat);
}
