#version 460

layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outColor;

layout (push_constant) uniform PushConstants {
    vec4 colorSeed;
} pc;

void main() {
    outColor = pc.colorSeed + vec4(inUV.x * 0.1, inUV.y * 0.1, 0.0, 1.0);
}
