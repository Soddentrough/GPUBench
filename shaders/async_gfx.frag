#version 460
layout(early_fragment_tests) in;

layout(location = 0) in vec2 vUV;
layout(location = 1) flat in uint vLayer;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConsts {
    uint layerCount;
    float seed;
} pc;

void main() {
    float layerPhase = float(vLayer) * (1.0 / float(max(pc.layerCount, 1u)));
    vec3 tint = fract(vec3(vUV.x * 0.37 + layerPhase + pc.seed,
                           vUV.y * 0.53 + layerPhase * 0.7 + pc.seed * 0.5,
                           (vUV.x + vUV.y) * 0.29 + layerPhase * 1.3));
    outColor = vec4(0.20 + 0.70 * tint, 1.0);
}
