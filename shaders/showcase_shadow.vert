#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "showcase_common.glsl"

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

layout(location = 0) out vec2 vUV;
layout(location = 1) flat out uint vMatId;
layout(location = 2) flat out uint vFlags;

void main() {
  ShowcaseVertex v = vertices[gl_VertexIndex];
  ShowcaseInstance inst = instances[gl_InstanceIndex];

  vec3 localPos = v.posMat.xyz;
  vec3 localNrm = v.normalFlags.xyz;
  uint vflags = floatBitsToUint(v.normalFlags.w) | inst.meta.w;
  if ((vflags & SHOWCASE_FLAG_TESSELLATED) != 0u) {
    localPos = showcaseDisplacement(localPos, localNrm);
  }

  vec4 worldPos = inst.model * vec4(localPos, 1.0);
  gl_Position = ubo.shadowViewProj * worldPos;
  vUV = v.uvPad.xy;
  vMatId = floatBitsToUint(v.posMat.w);
  vFlags = vflags;
}
