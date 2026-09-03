#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 attribs;

void main() {
    vec3 V = -normalize(gl_WorldRayDirectionEXT);
    vec3 N = normalize(vec3(attribs.x - 0.5, 1.0, attribs.y - 0.5));
    vec3 L = normalize(vec3(0.577, 0.577, 0.577));
    vec3 H = normalize(V + L);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float NdotV = clamp(dot(N, V), 0.001, 1.0);
    float NdotH = clamp(dot(N, H), 0.0, 1.0);
    float VdotH = clamp(dot(V, H), 0.0, 1.0);
    
    // GGX D
    float alpha = 0.25;
    float a2 = alpha * alpha;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (3.14159 * denom * denom);
    // Smith G
    float k = (alpha + 1.0) * (alpha + 1.0) / 8.0;
    float G = (NdotL / (NdotL * (1.0 - k) + k)) * (NdotV / (NdotV * (1.0 - k) + k));
    // Conductor Fresnel (Gold)
    vec3 F = vec3(1.0, 0.78, 0.34) + (1.0 - vec3(1.0, 0.78, 0.34)) * pow(1.0 - VdotH, 5.0);
    payload = (D * G * F) / (4.0 * NdotL * NdotV + 0.001);
}
