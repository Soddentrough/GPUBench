#version 460

layout(location = 0) out vec2 vUV;
layout(location = 1) flat out uint vLayer;

layout(push_constant) uniform PushConsts {
    uint layerCount;
    float seed;
} pc;

void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vUV = pos;
    uint layer = uint(gl_InstanceIndex);
    vLayer = layer;
    float z = 0.1 + 0.8 * (float(layer) / float(max(pc.layerCount, 1u)));
    gl_Position = vec4(pos * 2.0 - 1.0, z, 1.0);
}
