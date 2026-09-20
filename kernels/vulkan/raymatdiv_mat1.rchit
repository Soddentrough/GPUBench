#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 attribs;

void main() {
    vec3 V = -normalize(gl_WorldRayDirectionEXT);
    vec3 N = normalize(vec3(attribs.x - 0.5, 1.0, attribs.y - 0.5));
    float cosTheta = clamp(dot(V, N), 0.0, 1.0);
    float eta = 1.5;
    float r0 = (1.0 - eta) / (1.0 + eta);
    r0 = r0 * r0;
    float F = r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);
    vec3 refractDir = refract(-V, N, 1.0 / eta);
    vec3 reflectDir = reflect(-V, N);
    payload = mix(vec3(0.1, 0.4, 0.8) * abs(refractDir), vec3(0.9) * abs(reflectDir), F);
}
