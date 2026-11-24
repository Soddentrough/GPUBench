#version 460

layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outColor;

layout (push_constant) uniform PushConstants {
    vec4 colorSeed;
} pc;

uint hash2d(uvec2 p, uint seed) {
    uint h = (p.x * 0x9E3779B1u) ^ (p.y * 0x85EBCA77u) ^ (seed * 0xC2B2AE3Du);
    h ^= h >> 16u;
    h *= 0x7FEB352Du;
    h ^= h >> 15u;
    h *= 0x846CA68Bu;
    h ^= h >> 16u;
    return h;
}

void main() {
    uvec2 px = uvec2(gl_FragCoord.xy);
    uint seedBits = floatBitsToUint(pc.colorSeed.x) ^ (floatBitsToUint(pc.colorSeed.y) << 1u);
    uint h = hash2d(px, seedBits);
    vec3 rgb = vec3(
        0.15 + 0.70 * (float((h >>  0u) & 0x3FFu) / 1023.0),
        0.15 + 0.70 * (float((h >> 10u) & 0x3FFu) / 1023.0),
        0.15 + 0.70 * (float((h >> 20u) & 0x3FFu) / 1023.0)
    );
    outColor = vec4(rgb, pc.colorSeed.a);
}
