#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "showcase_common.glsl"

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

layout(location = 0) out vec3 vLocalPos;
layout(location = 1) out vec3 vLocalNormal;
layout(location = 2) out vec4 vLocalTangent;
layout(location = 3) out vec2 vUV;
layout(location = 4) flat out uint vMatId;
layout(location = 5) flat out uint vFlags;
layout(location = 6) flat out uint vInstanceIdx;

void main() {
  ShowcaseVertex v = vertices[gl_VertexIndex];
  ShowcaseInstance inst = instances[gl_InstanceIndex];
  vLocalPos = v.posMat.xyz;
  vLocalNormal = v.normalFlags.xyz;
  vLocalTangent = v.tangent;
  vUV = v.uvPad.xy;
  vMatId = floatBitsToUint(v.posMat.w);
  vFlags = floatBitsToUint(v.normalFlags.w) | inst.meta.w;
  vInstanceIdx = gl_InstanceIndex;
  gl_Position = vec4(vLocalPos, 1.0);
}
