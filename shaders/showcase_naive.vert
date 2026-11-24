#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "showcase_common.glsl"

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vWorldNormal;
layout(location = 2) out vec4 vWorldTangent;
layout(location = 3) out vec2 vUV;
layout(location = 4) flat out uint vMatId;
layout(location = 5) flat out uint vFlags;

void main() {
  ShowcaseVertex v = vertices[gl_VertexIndex];
  ShowcaseInstance inst = instances[gl_InstanceIndex];

  vec3 localPos = v.posMat.xyz;
  vec3 localNrm = v.normalFlags.xyz;
  vec3 localTan = v.tangent.xyz;
  uint flags = floatBitsToUint(v.normalFlags.w) | inst.meta.w;

  mat3 nrmMat = mat3(inst.model);
  vec4 wp = inst.model * vec4(localPos, 1.0);
  vWorldPos = wp.xyz;
  vWorldNormal = normalize(nrmMat * localNrm);
  vWorldTangent = vec4(normalize(nrmMat * localTan), v.tangent.w);
  vUV = v.uvPad.xy;
  vMatId = floatBitsToUint(v.posMat.w);
  vFlags = flags;

  gl_Position = ubo.viewProj * wp;
}
