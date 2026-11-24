#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "showcase_common.glsl"

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

layout(location = 0) out vec2 vUV;
layout(location = 1) flat out uint vMatId;
layout(location = 2) flat out uint vFlags;
layout(location = 3) flat out uvec2 vMetaIds; // x=meshletIdx, y=instanceIdx

void main() {
  uvec4 meta = fallbackCmds[gl_InstanceIndex * 2u + 1u];
  uint instIdx = meta.y;
  uint mIdx = meta.z;

  ShowcaseVertex v = vertices[gl_VertexIndex];
  ShowcaseInstance inst = instances[instIdx];
  vec4 wp = inst.model * vec4(v.posMat.xyz, 1.0);
  gl_Position = ubo.viewProj * wp;
  vUV = v.uvPad.xy;
  vMatId = floatBitsToUint(v.posMat.w);
  vFlags = floatBitsToUint(v.normalFlags.w) | inst.meta.w;
  vMetaIds = uvec2(mIdx, instIdx);
}
